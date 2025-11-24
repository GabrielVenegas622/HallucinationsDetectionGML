"""
Visualización de Trayectorias Latentes para Casos Específicos.

Este script carga un modelo GraphSequenceClassifier entrenado y busca 4 casos de uso
específicos en el dataset de validación para visualizar cómo evolucionan sus
representaciones latentes (μ) a lo largo del tiempo.

Casos de Uso:
1.  **True Positive (Alta Confianza):** Una alucinación que el modelo detecta con alta probabilidad.
2.  **True Positive (Baja Confianza):** Una alucinación que el modelo detecta con duda (cercano al umbral).
3.  **True Negative (Baja Confianza):** Un texto verídico que el modelo casi clasifica como alucinación.
4.  **True Negative (Alta Confianza):** Un texto verídico que el modelo clasifica correctamente con alta probabilidad.

El script realiza los siguientes pasos:
1.  Carga el modelo y los datos de validación.
2.  Itera sobre un subconjunto de los datos para encontrar los 4 casos de interés.
3.  Para cada caso, extrae la secuencia de vectores de media latente (μ) pasando los datos
    a través del encoder del modelo.
4.  Usa PCA para reducir la dimensionalidad de todas las secuencias a 2D.
5.  Grafica las 4 trayectorias, estilizadas por caso y confianza, con marcadores
    de inicio y fin.
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

# Asumiendo que DynGAD.py está en el mismo directorio o en el python path
from DynGAD import GraphSequenceClassifier, PreprocessedGNNDataset, collate_gnn_batch

# Silenciar avisos de matplotlib sobre fuentes
import logging
logging.getLogger('matplotlib.font_manager').disabled = True


def find_target_cases(model, data_loader, device, num_samples_to_check=200):
    """
    Busca en el data_loader los 4 casos de uso específicos.
    """
    model.eval() 
    
    # Estructura para almacenar el mejor candidato encontrado para cada caso
    # (distancia_al_target, probabilidad, índice_en_loader, datos)
    candidates = {
        'tp_clear': {'dist': float('inf'), 'prob': -1, 'idx': -1, 'data': None, 'label': -1},
        'tp_border': {'dist': float('inf'), 'prob': -1, 'idx': -1, 'data': None, 'label': -1},
        'tn_border': {'dist': float('inf'), 'prob': -1, 'idx': -1, 'data': None, 'label': -1},
        'tn_clear': {'dist': float('inf'), 'prob': -1, 'idx': -1, 'data': None, 'label': -1},
    }
    
    print(f"🔍 Buscando {num_samples_to_check} muestras para encontrar casos de uso...")
    with torch.no_grad():
        for i, (batched_by_layer, labels, _) in enumerate(tqdm(data_loader, total=num_samples_to_check)):
            if i >= num_samples_to_check:
                break
            
            # Procesar de a uno para mantener la correspondencia
            graph_seq, label = batched_by_layer, labels[0].item()
            
            graph_seq_gpu = [d.to(device) for d in graph_seq]
            logits, _ = model(graph_seq_gpu)
            prob = torch.sigmoid(logits).item()

            if label == 1: # Alucinación (True Positive)
                # TP Claro (target > 0.9)
                dist_clear = abs(prob - 0.95)
                if prob > 0.9 and dist_clear < candidates['tp_clear']['dist']:
                    candidates['tp_clear'] = {'dist': dist_clear, 'prob': prob, 'idx': i, 'data': graph_seq, 'label': label}

                # TP Dudoso (target ~0.55)
                dist_border = abs(prob - 0.55)
                if 0.45 < prob < 0.65 and dist_border < candidates['tp_border']['dist']:
                    candidates['tp_border'] = {'dist': dist_border, 'prob': prob, 'idx': i, 'data': graph_seq, 'label': label}

            else: # Verdad (True Negative)
                # TN Claro (target < 0.1)
                dist_clear = abs(prob - 0.05)
                if prob < 0.1 and dist_clear < candidates['tn_clear']['dist']:
                    candidates['tn_clear'] = {'dist': dist_clear, 'prob': prob, 'idx': i, 'data': graph_seq, 'label': label}
                
                # TN Dudoso (target ~0.45)
                dist_border = abs(prob - 0.45)
                if 0.35 < prob < 0.55 and dist_border < candidates['tn_border']['dist']:
                    candidates['tn_border'] = {'dist': dist_border, 'prob': prob, 'idx': i, 'data': graph_seq, 'label': label}

    print("\n--- Casos Encontrados ---")
    for name, c in candidates.items():
        if c['data']:
            print(f"  ✅ {name.upper()}: Muestra #{c['idx']}, Prob={c['prob']:.3f}, Label={c['label']}")
        else:
            print(f"  ❌ {name.upper()}: No se encontró un caso ideal.")
            
    return candidates

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
    Aplica PCA y grafica las trayectorias 2D.
    """
    print("🎨 Visualizando trayectorias con PCA...")
    
    all_mus = np.vstack([traj for traj in trajectories.values()])
    
    # Ajustar PCA con todos los puntos de todas las trayectorias
    pca = PCA(n_components=2)
    pca.fit(all_mus)
    
    # Transformar cada trayectoria individualmente
    transformed_trajs = {name: pca.transform(traj) for name, traj in trajectories.items()}
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    styles = {
        'tp_clear':  {'color': 'red', 'linestyle': '-', 'label': 'TP (Confianza Alta)', 'alpha': 1.0},
        'tp_border': {'color': 'red', 'linestyle': '--', 'label': 'TP (Borde)', 'alpha': 0.7},
        'tn_clear':  {'color': 'blue', 'linestyle': '-', 'label': 'TN (Confianza Alta)', 'alpha': 1.0},
        'tn_border': {'color': 'blue', 'linestyle': '--', 'label': 'TN (Borde)', 'alpha': 0.7},
    }

    for name, traj_2d in transformed_trajs.items():
        style = styles[name]
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], color=style['color'], linestyle=style['linestyle'], alpha=style['alpha'], label=style['label'])
        
        # Marcador de Inicio
        ax.scatter(traj_2d[0, 0], traj_2d[0, 1], marker='o', color=style['color'], s=100, edgecolors='black', zorder=5)
        ax.text(traj_2d[0, 0], traj_2d[0, 1] + 0.05, 'Inicio', fontsize=9, ha='center', color=style['color'])
        
        # Marcador de Fin
        ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], marker='X', color=style['color'], s=150, edgecolors='black', zorder=5)
        ax.text(traj_2d[-1, 0], traj_2d[-1, 1] - 0.15, 'Fin', fontsize=9, ha='center', color=style['color'])

    ax.set_title('Trayectorias en Espacio Latente (PCA) de Casos Específicos', fontsize=16)
    ax.set_xlabel('Componente Principal 1', fontsize=12)
    ax.set_ylabel('Componente Principal 2', fontsize=12)
    ax.legend(title="Leyenda de Casos", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
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
    # Usaremos una porción para validación
    _ , val_files, _ = np.split(all_files, [int(0.7*len(all_files)), int(0.85*len(all_files))])

    # El dataset para buscar casos (de a uno)
    val_dataset = PreprocessedGNNDataset(gnn_dir, val_files)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, collate_fn=collate_gnn_batch)
    
    # Para obtener hidden_dim, necesitamos cargar al menos un dato
    temp_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, collate_fn=collate_gnn_batch)
    sample_data = next(iter(temp_loader))
    hidden_dim = sample_data[0][0].x.shape[-1]
    del temp_loader, sample_data; gc.collect()

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
    if trajectories:
        visualize_trajectories(trajectories, target_cases, args.output_file)
    else:
        print("No se encontraron trayectorias para visualizar. Terminando.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualizar trayectorias latentes de casos específicos.")
    parser.add_argument('--model-path', type=str, default='dyngad_results/best_model.pt', help='Ruta al modelo entrenado (.pt)')
    parser.add_argument('--preprocessed-dir', type=str, default='preprocessed_data', help='Directorio con datos preprocesados ("gnn")')
    parser.add_argument('--output-file', type=str, default='visualizations/latent_trajectories_cases.png', help='Ruta para guardar el gráfico.')
    parser.add_argument('--num-samples', type=int, default=200, help='Número de muestras a revisar para encontrar los casos.')
    # Añadir argumentos del modelo para poder instanciarlo
    parser.add_argument('--gnn-hidden', type=int, default=128)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--lstm-hidden', type=int, default=32)
    parser.add_argument('--num-lstm-layers', type=int, default=2)
    parser.add_argument('--num-clusters', type=int, default=32)
    parser.add_argument('--force-cpu', action='store_true')

    args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    main(args)
