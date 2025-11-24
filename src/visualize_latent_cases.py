"""
Visualización de Trayectorias Latentes para Casos Específicos.

Este script carga un modelo GraphSequenceClassifier entrenado, busca 4 casos de uso
específicos (2 alucinaciones de alta confianza y 2 verdades de alta confianza)
en el dataset de validación para visualizar y comparar sus trayectorias latentes.

El script realiza los siguientes pasos:
1.  Carga el modelo y los datos de validación.
2.  Itera sobre un subconjunto de los datos para encontrar los casos de interés.
3.  Selecciona aleatoriamente 2 casos de cada categoría (TP y TN).
4.  Para cada caso, extrae la secuencia de vectores de media latente (μ).
5.  Usa PCA para reducir la dimensionalidad a 2D.
6.  Grafica las 4 trayectorias, aplicando un estilo similar al de `visualize_baseline.py`,
    mostrando cada estado como un punto en la trayectoria y eliminando los números de los ejes.
"""
import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import gc
import torch.nn.functional as F # Importar F para relu
import random

# Asumiendo que DynGAD.py está en el mismo directorio o en el python path
from DynGAD import GraphSequenceClassifier, PreprocessedGNNDataset, collate_gnn_batch

# Silenciar avisos de matplotlib sobre fuentes
import logging
logging.getLogger('matplotlib.font_manager').disabled = True


def find_target_cases(model, data_loader, device, num_samples_to_check=200):
    """
    Busca y selecciona aleatoriamente 2 casos de alucinaciones claras y 2 de verdades claras.
    """
    model.eval()
    
    # Listas para almacenar todos los candidatos encontrados
    tp_candidates = []
    tn_candidates = []
    
    print(f"🔍 Buscando en hasta {num_samples_to_check} muestras para encontrar 2 TPs y 2 TNs de alta confianza...")
    with torch.no_grad():
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

            if label == 1 and prob > 0.9: # Alucinación (True Positive) de alta confianza
                tp_candidates.append({'prob': prob, 'data': graph_seq, 'label': label})

            elif label == 0 and prob < 0.1: # Verdad (True Negative) de alta confianza
                tn_candidates.append({'prob': prob, 'data': graph_seq, 'label': label})

    # Seleccionar aleatoriamente 2 de cada categoría si hay suficientes
    final_cases = {}
    if len(tp_candidates) >= 2:
        selected_tps = random.sample(tp_candidates, 2)
        final_cases['tp_clear_1'] = selected_tps[0]
        final_cases['tp_clear_2'] = selected_tps[1]
    else:
        print(f"⚠️ No se encontraron suficientes casos de alucinaciones de alta confianza (encontrados: {len(tp_candidates)}).")

    if len(tn_candidates) >= 2:
        selected_tns = random.sample(tn_candidates, 2)
        final_cases['tn_clear_1'] = selected_tns[0]
        final_cases['tn_clear_2'] = selected_tns[1]
    else:
        print(f"⚠️ No se encontraron suficientes casos de verdad de alta confianza (encontrados: {len(tn_candidates)}).")

    print("\n--- Casos Seleccionados Aleatoriamente ---")
    for name, c in final_cases.items():
        print(f"  ✅ {name.upper()}: Prob={c['prob']:.3f}, Label={c['label']}")
    if len(final_cases) < 4:
        print("  ❌ No se graficará por no tener suficientes casos.")
            
    return final_cases

def extract_latent_trajectories(model, target_cases, device):
    """
    Pasa los grafos seleccionados por el encoder para extraer las secuencias de μ.
    """
    model.eval()
    trajectories = {}
    
    print("\n🔬 Extrayendo trayectorias latentes...")
    with torch.no_grad():
        for name, case in tqdm(target_cases.items()):
            if case['data'] is None:
                continue

            mus = []
            graph_seq_gpu = [d.to(device) for d in case['data']]
            
            for layer_data in graph_seq_gpu:
                x, edge_index, edge_attr, batch = layer_data.x, layer_data.edge_index, layer_data.edge_attr, layer_data.batch
                x = x.float() # Asegurar que x sea float32
                
                # --- Bloque de Saneamiento de Datos (Copiado de DynGAD.py) ---
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

                # Replicando el forward del encoder
                x = model.input_proj(x)
                x_gnn = F.relu(model.bn1(model.conv1(x, edge_index, edge_attr)))
                x_gnn = model.bn2(model.conv2(x_gnn, edge_index, edge_attr))
                
                x_dense, mask = torch_geometric.utils.to_dense_batch(x_gnn, batch, max_num_nodes=200)
                adj = torch_geometric.utils.to_dense_adj(edge_index, batch, max_num_nodes=200)
                
                s = model.pool_mlp(x_dense)
                x_pooled, _, _, _ = torch_geometric.nn.dense.dense_mincut_pool(x_dense, adj, s, mask)
                
                graph_repr = x_pooled.flatten(start_dim=1)
                graph_repr = model.pre_vae_bn(graph_repr)
                mu = model.fc_mu(graph_repr)
                mus.append(mu.cpu().numpy())
            
            trajectories[name] = np.vstack(mus)
            
    return trajectories


def visualize_trajectories(trajectories, cases_info, output_path):
    """
    Aplica PCA y grafica las trayectorias 2D con el nuevo estilo.
    """
    if not trajectories:
        print("No hay trayectorias para visualizar.")
        return

    print("🎨 Visualizando trayectorias con PCA...")
    
    all_mus = np.vstack([traj for traj in trajectories.values()])
    
    # Ajustar PCA con todos los puntos de todas las trayectorias
    pca = PCA(n_components=2)
    pca.fit(all_mus)
    
    # Transformar cada trayectoria individualmente
    transformed_trajs = {name: pca.transform(traj) for name, traj in trajectories.items()}
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Paleta de colores y estilos
    colors = {'tp': '#d62728', 'tn': '#1f77b4'} # Rojo para TP, Azul para TN
    markers = ['o', 's']
    alphas = [1.0, 0.7]

    # Contadores para asignar estilos
    tp_count = 0
    tn_count = 0

    for name in sorted(transformed_trajs.keys()):
        traj_2d = transformed_trajs[name]
        
        if 'tp' in name:
            case_type = 'tp'
            style_idx = tp_count
            label = f'Alucinación {style_idx+1}'
            tp_count += 1
        else:
            case_type = 'tn'
            style_idx = tn_count
            label = f'Verdad {style_idx+1}'
            tn_count += 1

        color = colors[case_type]
        marker = markers[style_idx]
        alpha = alphas[style_idx]

        # Plot de la trayectoria con marcadores en cada punto (estado)
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], 
                color=color, 
                alpha=alpha, 
                label=label,
                linewidth=2,
                marker=marker,
                markersize=6,
                linestyle='-')
        
        # Marcador de Inicio
        ax.scatter(traj_2d[0, 0], traj_2d[0, 1], marker='o', s=150, facecolors='none', edgecolors=color, linewidth=2, zorder=5)
        
        # Marcador de Fin
        ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], marker='X', color=color, s=150, edgecolors='black', zorder=5)

    ax.set_title('Evolución de Trayectorias en Espacio Latente (PCA)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Componente Principal 1', fontsize=12)
    ax.set_ylabel('Componente Principal 2', fontsize=12)
    
    # Eliminar números de los ejes
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.legend(title="Casos de Alta Confianza", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n✅ Gráfico guardado en: {output_path}")
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Dispositivo: {device}")

    # --- Carga de Datos ---
    gnn_dir = Path(args.preprocessed_dir) / 'gnn'
    all_files = sorted(list(gnn_dir.glob('preprocessed_*.pt')))
    
    # FIX: Usar slicing en lugar de np.split para evitar seleccionar un solo archivo
    train_idx = int(len(all_files) * 0.7)
    val_idx = int(len(all_files) * 0.85)
    val_files = all_files[train_idx:val_idx]

    if not val_files:
        print("Error: No se encontraron archivos de validación con el split 70/15/15.")
        return

    val_dataset = PreprocessedGNNDataset(gnn_dir, list(val_files))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, collate_fn=collate_gnn_batch, num_workers=0)
    
    # Para obtener hidden_dim, necesitamos cargar al menos un dato
    try:
        # Usamos el primer archivo de validación para eficiencia
        temp_loader = torch.utils.data.DataLoader(PreprocessedGNNDataset(gnn_dir, [val_files[0]]), batch_size=1, collate_fn=collate_gnn_batch)
        sample_data, _, _ = next(iter(temp_loader))
        hidden_dim = sample_data[0][0].x.shape[-1]
        del temp_loader, sample_data
        gc.collect()
    except (StopIteration, IndexError):
        print("Error: No se pudieron cargar datos de validación para determinar la dimensión de entrada.")
        return

    # --- Carga de Modelo ---
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
        
    model = GraphSequenceClassifier(hidden_dim=hidden_dim, gnn_hidden=args.gnn_hidden, latent_dim=args.latent_dim,
                                    lstm_hidden=args.lstm_hidden, num_lstm_layers=args.num_lstm_layers,
                                    num_clusters=args.num_clusters).float()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Modelo cargado exitosamente.")

    # 1. Buscar casos
    target_cases = find_target_cases(model, val_loader, device, num_samples_to_check=args.num_samples)
    
    # 2. Extraer trayectorias
    # Necesitamos importar torch_geometric dinámicamente para el forward manual
    global torch_geometric
    import torch_geometric
    trajectories = extract_latent_trajectories(model, target_cases, device)
    
    # 3. Visualizar
    visualize_trajectories(trajectories, target_cases, args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualizar trayectorias latentes de casos específicos.")
    parser.add_argument('--model-path', type=str, default='dyngad_results/best_model.pt', help='Ruta al modelo entrenado (.pt)')
    parser.add_argument('--preprocessed-dir', type=str, default='preprocessed_data', help='Directorio con datos preprocesados ("gnn")')
    parser.add_argument('--output-file', type=str, default='visualizations/latent_trajectories_cases.png', help='Ruta para guardar el gráfico.')
    parser.add_argument('--num-samples', type=int, default=200, help='Número de muestras a revisar para encontrar los casos.')
    # Añadir argumentos del modelo para poder instanciarlo
    parser.add_argument('--gnn-hidden', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--lstm-hidden', type=int, default=32)
    parser.add_argument('--num-lstm-layers', type=int, default=2)
    parser.add_argument('--num-clusters', type=int, default=32)
    parser.add_argument('--force-cpu', action='store_true')

    args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    main(args)
