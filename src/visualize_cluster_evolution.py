"""
Visualización de la Evolución de Clusters Latentes para Casos Específicos.

Este script carga un modelo GraphSequenceClassifier entrenado, busca dos casos
específicos (una alucinación de alta confianza y una verdad de alta confianza)
y visualiza la evolución de la estructura de su cluster latente a través de
las capas del LLM.

Proceso:
1.  Carga un checkpoint del modelo entrenado y el dataset de validación.
2.  Busca en los datos una "alucinación clara" (label=1, prob>0.9) y una
    "verdad clara" (label=0, prob<0.1).
3.  Para cada uno de los dos casos, itera sobre un conjunto predefinido de capas.
4.  En cada capa, extrae el vector `mu` y reconstruye la matriz de adyacencia
    de los clusters (A = sigmoid(μ * μ^T)).
5.  Opcionalmente, aplica un umbral a la matriz para filtrar conexiones débiles.
6.  Utiliza una función de ploteo externa para dibujar las matrices,
    agrupándolas en imágenes y guardándolas con nombres descriptivos.
"""
import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# Importaciones del proyecto
from DynGAD import GraphSequenceClassifier, PreprocessedGNNDataset, collate_gnn_batch

# --- IMPORTACIÓN CRÍTICA ---
# Asumimos que el script `visualize_attention_graph.py` contiene una función
# para plotear. Ajusta el nombre de la función si es necesario.
# La firma ideal sería: `plot_adjacency_matrix(adj, title, ax)` para que
# pueda dibujar sobre un subplot de matplotlib.
try:
    from visualize_attention_graph import plot_adjacency_matrix
except ImportError:
    print("="*80)
    print("ERROR: No se pudo importar `plot_adjacency_matrix` desde `src/visualize_attention_graph.py`.")
    print("Creando una función placeholder. Por favor, reemplázala o corrige la importación.")
    print("="*80)
    def plot_adjacency_matrix(adj, title, ax):
        ax.imshow(adj, cmap='hot', interpolation='nearest')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

# Silenciar avisos de matplotlib
import logging
logging.getLogger('matplotlib.font_manager').disabled = True


def find_comparison_cases(model, data_loader, device, num_samples_to_check):
    """
    Busca en el data_loader una alucinación clara y una verdad clara.
    """
    model.eval()
    
    candidates = {
        'hallucination': {'dist': float('inf'), 'prob': -1, 'data': None, 'label': -1},
        'truth': {'dist': float('inf'), 'prob': -1, 'data': None, 'label': -1},
    }
    
    print(f"🔍 Buscando en {num_samples_to_check} muestras para encontrar casos de comparación...")
    with torch.no_grad():
        for i, (batched_by_layer, labels, _) in enumerate(tqdm(data_loader, total=num_samples_to_check)):
            if i >= num_samples_to_check:
                break
            
            graph_seq, label = batched_by_layer, labels[0].item()
            
            graph_seq_gpu = [d.to(device) for d in graph_seq]
            logits, _ = model(graph_seq_gpu)
            prob = torch.sigmoid(logits).item()

            if label == 1: # Alucinación (True Positive)
                dist = abs(prob - 0.95)
                if prob > 0.9 and dist < candidates['hallucination']['dist']:
                    candidates['hallucination'] = {'dist': dist, 'prob': prob, 'data': graph_seq, 'label': label}
            else: # Verdad (True Negative)
                dist = abs(prob - 0.05)
                if prob < 0.1 and dist < candidates['truth']['dist']:
                    candidates['truth'] = {'dist': dist, 'prob': prob, 'data': graph_seq, 'label': label}
    
    print("\n--- Casos Encontrados para Comparación ---")
    for name, c in candidates.items():
        if c['data']:
            print(f"  ✅ {name.capitalize()}: Prob={c['prob']:.3f}, Label={c['label']}")
        else:
            print(f"  ❌ {name.capitalize()}: No se encontró un caso ideal.")
            
    return {k: v for k, v in candidates.items() if v['data'] is not None}

def extract_and_reconstruct_adj(model, graph_seq, device, num_clusters, latent_dim):
    """
    Realiza el forward pass manual y reconstruye la matriz de adyacencia.
    """
    model.eval()
    with torch.no_grad():
        layer_data = graph_seq.to(device)
        x, edge_index, edge_attr, batch = layer_data.x, layer_data.edge_index, layer_data.edge_attr, layer_data.batch

        # --- Bloque de Saneamiento de Datos ---
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = edge_attr.float()
            if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1.0, neginf=0.0)
            
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            
            if edge_attr.size(0) != edge_index.size(1):
                num_edges = edge_index.size(1)
                if edge_attr.size(0) > num_edges:
                    edge_attr = edge_attr[:num_edges]
                else:
                    padding = torch.zeros((num_edges - edge_attr.size(0), 1), 
                                        dtype=edge_attr.dtype, device=edge_attr.device)
                    edge_attr = torch.cat([edge_attr, padding], dim=0)
            
            edge_attr = torch.clamp(edge_attr, min=0.0, max=1.0)
        else:
            num_edges = edge_index.size(1)
            edge_attr = torch.zeros((num_edges, 1), dtype=torch.float, device=x.device)
        # --- Fin del Bloque de Saneamiento ---

        # 1. Forward pass del Encoder GINE
        x = model.input_proj(x)
        x_gnn = torch.nn.functional.relu(model.bn1(model.conv1(x, edge_index, edge_attr)))
        x_gnn = model.bn2(model.conv2(x_gnn, edge_index, edge_attr))
        
        # 2. MincutPool
        x_dense, mask = torch_geometric.utils.to_dense_batch(x_gnn, batch, max_num_nodes=200)
        adj_dense = torch_geometric.utils.to_dense_adj(edge_index, batch, max_num_nodes=200)
        
        s = model.pool_mlp(x_dense)
        x_pooled, _, _, _ = torch_geometric.nn.dense.dense_mincut_pool(x_dense, adj_dense, s, mask)
        
        # 3. Proyección a `mu`
        graph_repr = x_pooled.flatten(start_dim=1)
        graph_repr = model.pre_vae_bn(graph_repr)
        mu = model.fc_mu(graph_repr)
        
        # 4. Decodificación por producto interno
        if latent_dim % num_clusters != 0:
            raise ValueError("latent_dim debe ser divisible por num_clusters")
        
        features_per_cluster = latent_dim // num_clusters
        mu_reshaped = mu.view(-1, num_clusters, features_per_cluster)
        
        adj_recons = torch.bmm(mu_reshaped, mu_reshaped.transpose(1, 2))
        adj_recons = torch.sigmoid(adj_recons)
        
    return adj_recons.squeeze(0).cpu().numpy()

def process_and_visualize_case(model, case_name, case_data, args, device):
    """
    Procesa una secuencia de grafos y genera las visualizaciones agrupadas.
    """
    print(f"\n--- Procesando caso: {case_name.upper()} ---")
    full_sequence_data = case_data['data']
    
    print("Extrayendo y reconstruyendo matrices de adyacencia latente...")
    reconstructed_adjs = {}
    layers_to_visualize = sorted(list(set(args.layers)))

    for layer_idx in tqdm(layers_to_visualize):
        if layer_idx >= len(full_sequence_data):
            continue
        
        graph_for_layer = full_sequence_data[layer_idx]
        adj_recons = extract_and_reconstruct_adj(model, graph_for_layer, device, args.num_clusters, args.latent_dim)
        
        if not args.no_threshold:
            adj_processed = (adj_recons > args.threshold).astype(float)
            np.fill_diagonal(adj_processed, 0)
        else:
            adj_processed = adj_recons

        reconstructed_adjs[layer_idx] = adj_processed

    # --- Bucle de Visualización ---
    print("Generando visualizaciones agrupadas...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layer_groups = [[0, 4, 8], [12, 16, 20], [24, 28, 31]]
    
    threshold_str = "no-threshold" if args.no_threshold else f"threshold-{args.threshold}"

    for i, group in enumerate(layer_groups):
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        main_title = f'Evolución de Clusters ({case_name.capitalize()}) - Capas {group[0]}-{group[-1]}'
        if not args.no_threshold:
             main_title += f' (Threshold > {args.threshold})'
        fig.suptitle(main_title, fontsize=20, y=1.02)
        
        for ax_idx, layer_idx in enumerate(group):
            ax = axes[ax_idx]
            if layer_idx in reconstructed_adjs:
                adj_matrix = reconstructed_adjs[layer_idx]
                title = f'Capa {layer_idx}'
                plot_adjacency_matrix(adj_matrix, title, ax=ax)
            else:
                ax.set_title(f'Capa {layer_idx} (No procesada)')
                ax.set_visible(False)

        plt.tight_layout()
        output_path = output_dir / f'cluster_evo_{case_name}_{threshold_str}_layers_{group[0]}-{group[-1]}.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en: {output_path}")
        plt.close(fig)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Dispositivo: {device}")
    
    # --- Carga de Datos ---
    gnn_dir = Path(args.preprocessed_dir) / 'gnn'
    all_files = sorted(list(gnn_dir.glob('preprocessed_*.pt')))
    _, val_files, _ = np.split(all_files, [int(0.7*len(all_files)), int(0.85*len(all_files))])
    
    val_dataset = PreprocessedGNNDataset(gnn_dir, val_files)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, collate_fn=collate_gnn_batch, num_workers=0)
    
    # Cargar un dato para obtener hidden_dim
    temp_loader = torch.utils.data.DataLoader(PreprocessedGNNDataset(gnn_dir, [val_files[0]]), batch_size=1, collate_fn=collate_gnn_batch)
    sample_data, _, _ = next(iter(temp_loader))
    hidden_dim = sample_data[0].x.shape[-1]
    del temp_loader, sample_data; gc.collect()

    # --- Carga de Modelo ---
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
        
    model = GraphSequenceClassifier(
        hidden_dim=hidden_dim, gnn_hidden=args.gnn_hidden, latent_dim=args.latent_dim,
        lstm_hidden=args.lstm_hidden, num_lstm_layers=args.num_lstm_layers,
        num_clusters=args.num_clusters, dropout=args.dropout
    ).float()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Modelo cargado exitosamente.")

    # Necesitamos importar torch_geometric para el forward manual
    global torch_geometric
    import torch_geometric

    # --- Búsqueda de Casos y Visualización ---
    comparison_cases = find_comparison_cases(model, val_loader, device, args.num_samples)
    
    if not comparison_cases:
        print("No se encontraron casos adecuados para la visualización. Terminando.")
        return

    for case_name, case_data in comparison_cases.items():
        process_and_visualize_case(model, case_name, case_data, args, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualizar la evolución de clusters latentes de DynGAD para casos específicos.")
    parser.add_argument('--model-path', type=str, default='dyngad_results/best_model.pt', help='Ruta al modelo entrenado (.pt)')
    parser.add_argument('--preprocessed-dir', type=str, default='preprocessed_data', help='Directorio con datos preprocesados ("gnn")')
    parser.add_argument('--output-dir', type=str, default='visualizations/cluster_evolution', help='Directorio para guardar los gráficos.')
    
    # Hiperparámetros del modelo
    parser.add_argument('--gnn-hidden', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--lstm-hidden', type=int, default=32)
    parser.add_argument('--num-lstm-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num-clusters', type=int, default=32)

    # Configuración de visualización y búsqueda
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 4, 8, 12, 16, 20, 24, 28, 31], help='Lista de índices de capas a visualizar.')
    parser.add_argument('--threshold', type=float, default=0.7, help='Umbral para binarizar la matriz de adyacencia reconstruida.')
    parser.add_argument('--no-threshold', action='store_true', help='Si se establece, no se aplica ningún umbral a la matriz de adyacencia.')
    parser.add_argument('--num-samples', type=int, default=200, help='Número de muestras a revisar para encontrar los casos.')
    
    parser.add_argument('--force-cpu', action='store_true')

    args = parser.parse_args()
    main(args)
