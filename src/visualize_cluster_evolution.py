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
6.  Utiliza `graph-tool` para dibujar las matrices de adyacencia resultantes,
    agrupándolas en imágenes y guardándolas con nombres descriptivos.
"""
import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import gc
import os
from PIL import Image

# Importaciones del proyecto
from DynGAD import GraphSequenceClassifier, PreprocessedGNNDataset, collate_gnn_batch

# --- Dependencia de graph-tool ---
try:
    import graph_tool.all as gt
    GRAPH_TOOL_AVAILABLE = True
except ImportError:
    print("="*80)
    print("ERROR CRÍTICO: El paquete 'graph-tool' no está instalado.")
    print("Esta visualización depende de graph-tool para generar los grafos.")
    print("Por favor, instálalo siguiendo las instrucciones en: https://graph-tool.skewed.de/installation")
    print("Ejemplo con conda: conda install -c conda-forge graph-tool")
    print("="*80)
    GRAPH_TOOL_AVAILABLE = False

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
        # Asegurarse de que el dataloader itere sobre todas las muestras necesarias
        for i, data in enumerate(tqdm(data_loader, total=num_samples_to_check, desc="Buscando casos")):
            if i >= num_samples_to_check:
                break
            if not data: continue
            
            batched_by_layer, labels, _ = data
            if not batched_by_layer or not labels.numel(): continue

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
        x = x.float()

        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = edge_attr.float()
            if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1.0, neginf=0.0)
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            if edge_attr.size(0) != edge_index.size(1):
                padding = torch.zeros((edge_index.size(1) - edge_attr.size(0), 1), dtype=edge_attr.dtype, device=edge_attr.device)
                edge_attr = torch.cat([edge_attr, padding], dim=0)
            edge_attr = torch.clamp(edge_attr, min=0.0, max=1.0)
        else:
            edge_attr = torch.zeros((edge_index.size(1), 1), dtype=torch.float, device=x.device)

        x = model.input_proj(x)
        x_gnn = torch.nn.functional.relu(model.bn1(model.conv1(x, edge_index, edge_attr)))
        x_gnn = model.bn2(model.conv2(x_gnn, edge_index, edge_attr))
        
        # This part requires torch_geometric, which should be loaded in main
        x_dense, mask = torch_geometric.utils.to_dense_batch(x_gnn, batch, max_num_nodes=200)
        adj_dense = torch_geometric.utils.to_dense_adj(edge_index, batch, max_num_nodes=200)
        
        s = model.pool_mlp(x_dense)
        x_pooled, _, _, _ = torch_geometric.nn.dense.dense_mincut_pool(x_dense, adj_dense, s, mask)
        
        graph_repr = x_pooled.flatten(start_dim=1)
        graph_repr = model.pre_vae_bn(graph_repr)
        mu = model.fc_mu(graph_repr)
        
        features_per_cluster = latent_dim // num_clusters
        mu_reshaped = mu.view(-1, num_clusters, features_per_cluster)
        
        adj_recons = torch.bmm(mu_reshaped, mu_reshaped.transpose(1, 2))
        adj_recons = torch.sigmoid(adj_recons)
        
    return adj_recons.squeeze(0).cpu().numpy()

def plot_cluster_graph(adj_matrix, output_path, is_binary):
    """
    Dibuja un grafo de clusters usando graph-tool y lo guarda en un archivo.
    """
    if not GRAPH_TOOL_AVAILABLE:
        return

    g = gt.Graph(directed=False)
    num_nodes = adj_matrix.shape[0]
    g.add_vertex(num_nodes)

    edge_weights = g.new_edge_property("double")
    
    rows, cols = np.where(adj_matrix > 0)
    for i in range(len(rows)):
        if rows[i] < cols[i]:
            e = g.add_edge(rows[i], cols[i])
            edge_weights[e] = adj_matrix[rows[i], cols[i]]

    pos = gt.sfdp_layout(g, cooling_step=0.95, epsilon=1e-3)

    vertex_fill_color = "#1f77b4"
    
    edge_pen_width = g.new_edge_property("double")
    edge_color = g.new_edge_property("vector<double>")
    viridis_cmap = cm.get_cmap('viridis')

    weights = np.array([edge_weights[e] for e in g.edges()])
    min_w, max_w = (0, 1) if is_binary else (np.min(weights), np.max(weights)) if len(weights) > 0 else (0, 1)

    for e in g.edges():
        weight = edge_weights[e]
        if is_binary:
            norm_weight = 0.8
            edge_pen_width[e] = 2.5
        else:
            norm_weight = (weight - min_w) / (max_w - min_w) if max_w > min_w else 0
            edge_pen_width[e] = 1.0 + norm_weight * 5.0
        
        rgba = viridis_cmap(norm_weight)
        edge_color[e] = list(rgba)
    
    gt.graph_draw(
        g, pos,
        vertex_fill_color=vertex_fill_color, vertex_size=25,
        edge_pen_width=edge_pen_width, edge_color=edge_color,
        bg_color=[1, 1, 1, 1], output_size=(1000, 1000),
        output=str(output_path)
    )

def process_and_visualize_case(model, case_name, case_data, args, device):
    """
    Procesa una secuencia de grafos y genera las visualizaciones agrupadas.
    """
    if not GRAPH_TOOL_AVAILABLE:
        print("Saltando visualización por falta de graph-tool.")
        return

    print(f"\n--- Procesando caso: {case_name.upper()} ---")
    full_sequence_data = case_data['data']
    
    reconstructed_adjs = {}
    layers_to_visualize = sorted(list(set(args.layers)))

    for layer_idx in tqdm(layers_to_visualize, desc=f"Extrayendo para {case_name}"):
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layer_groups = [[0, 4, 8], [12, 16, 20], [24, 28, 31]]
    threshold_str = "no-threshold" if args.no_threshold else f"threshold-{args.threshold}"

    for i, group in enumerate(layer_groups):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        main_title = f'Evolución de Clusters ({case_name.capitalize()}) - Capas {group[0]}-{group[-1]}'
        if not args.no_threshold:
             main_title += f' (Threshold > {args.threshold})'
        fig.suptitle(main_title, fontsize=18, y=1.0)
        
        temp_files = []
        for ax_idx, layer_idx in enumerate(group):
            ax = axes[ax_idx]
            ax.axis('off')
            ax.set_title(f'Capa {layer_idx}', fontsize=14)
            if layer_idx in reconstructed_adjs:
                adj_matrix = reconstructed_adjs[layer_idx]
                temp_file = output_dir / f"temp_{case_name}_{layer_idx}.png"
                temp_files.append(temp_file)
                
                plot_cluster_graph(adj_matrix, temp_file, is_binary=not args.no_threshold)
                
                if os.path.exists(temp_file):
                    img = Image.open(temp_file)
                    ax.imshow(img)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = output_dir / f'cluster_evo_{case_name}_{threshold_str}_layers_{group[0]}-{group[-1]}.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en: {output_path}")
        plt.close(fig)

        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Dispositivo: {device}")
    
    gnn_dir = Path(args.preprocessed_dir) / 'gnn'
    all_files = sorted(list(gnn_dir.glob('preprocessed_*.pt')))
    train_idx = int(len(all_files) * 0.7)
    val_idx = int(len(all_files) * 0.85)
    val_files = all_files[train_idx:val_idx]
    
    if not val_files:
        print("Error: No se encontraron archivos de validación.")
        return

    val_dataset = PreprocessedGNNDataset(gnn_dir, val_files)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, collate_fn=collate_gnn_batch, num_workers=0)
    
    try:
        temp_loader = torch.utils.data.DataLoader(PreprocessedGNNDataset(gnn_dir, [val_files[0]]), batch_size=1, collate_fn=collate_gnn_batch)
        sample_data, _, _ = next(iter(temp_loader))
        hidden_dim = sample_data[0][0].x.shape[-1]
        del temp_loader, sample_data; gc.collect()
    except (StopIteration, IndexError):
        print("Error: No se pudieron cargar datos de validación para determinar la dimensión de entrada.")
        return

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

    global torch_geometric
    import torch_geometric

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
    
    parser.add_argument('--gnn-hidden', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--lstm-hidden', type=int, default=32)
    parser.add_argument('--num-lstm-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num-clusters', type=int, default=32)

    parser.add_argument('--layers', type=int, nargs='+', default=[0, 4, 8, 12, 16, 20, 24, 28, 31], help='Lista de índices de capas a visualizar.')
    parser.add_argument('--threshold', type=float, default=0.7, help='Umbral para binarizar la matriz de adyacencia reconstruida.')
    parser.add_argument('--no-threshold', action='store_true', help='Si se establece, no se aplica ningún umbral a la matriz de adyacencia.')
    parser.add_argument('--num-samples', type=int, default=200, help='Número de muestras a revisar para encontrar los casos.')
    
    parser.add_argument('--force-cpu', action='store_true')

    args = parser.parse_args()
    if GRAPH_TOOL_AVAILABLE:
        main(args)
    else:
        print("El script no puede continuar porque 'graph-tool' no está instalado.")
