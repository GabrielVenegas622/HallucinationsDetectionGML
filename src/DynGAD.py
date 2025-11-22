"""
Entrenamiento de un clasificador de secuencias de grafos GINE-VAE-LSTM.

Este script implementa y entrena la arquitectura `GraphSequenceClassifier` para
la clasificación binaria de secuencias de grafos.

ARQUITECTURA:
1.  **Encoder Estructural (GINE + Pooling Jerárquico):**
    -   Capas GINEConv procesan las características de nodos y aristas.
    -   Los datos dispersos (sparse) se convierten a formato denso.
    -   `dense_mincut_pool` agrupa los nodos en 32 clusters, generando una
        representación de alto nivel y dos pérdidas auxiliares (Mincut, Orthogonality).
    -   La salida aplanada del pooling se proyecta a los parámetros de una
        distribución latente: `mu` y `log_std` (dimensión 128 cada uno).

2.  **Componente VAE (Regularización):**
    -   Se implementa el "reparameterization trick" para muestrear `z`.
    -   Se calcula una `aux_loss` que incluye:
        -   `mincut_loss` y `orthogonality_loss` del pooling.
        -   Divergencia KL entre la distribución latente y una N(0,1).
        -   Pérdida de reconstrucción de la matriz de adyacencia densa y agrupada.
    -   Esta `aux_loss` se suma a la pérdida de la tarea principal.

3.  **Procesamiento Temporal (LSTM):**
    -   La entrada al LSTM es determinista: `torch.cat([mu, log_std])`. Esto
        separa la regularización estocástica de la dinámica temporal.
    -   Un LSTM procesa la secuencia de estas representaciones.

4.  **Clasificador de Salida:**
    -   Una capa lineal toma el último estado oculto del LSTM para producir
        logits crudos, aptos para `BCEWithLogitsLoss`.

Uso:
    python src/DynGAD.py --preprocessed-dir preprocessed_data --epochs 50
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn.dense import dense_mincut_pool
from torch_geometric.data import Batch as PyGBatch
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import random
import multiprocessing
import gc
from collections import deque

# ============================================================================
# ARQUITECTURA PRINCIPAL: GraphSequenceClassifier
# ============================================================================

class GraphSequenceClassifier(torch.nn.Module):
    """
    Arquitectura GINE-VAE-LSTM para clasificación binaria de secuencias de grafos.
    """
    def __init__(self, hidden_dim, gnn_hidden=128, latent_dim=128, lstm_hidden=256, num_lstm_layers=2, dropout=0.3):
        super().__init__()
        
        self.gnn_hidden = gnn_hidden
        self.latent_dim = latent_dim
        self.num_clusters = 32  # Especificación: 32 clusters para MincutPool

        # Componente 1: Encoder Estructural (GINE)
        self.conv1 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, gnn_hidden), nn.ReLU(), nn.Linear(gnn_hidden, gnn_hidden)),
            edge_dim=1
        )
        self.bn1 = nn.BatchNorm1d(gnn_hidden)
        self.conv2 = GINEConv(
            nn.Sequential(nn.Linear(gnn_hidden, gnn_hidden), nn.ReLU(), nn.Linear(gnn_hidden, gnn_hidden)),
            edge_dim=1
        )
        self.bn2 = nn.BatchNorm1d(gnn_hidden)

        # Componente 1: MLP para aprender la asignación de clusters en MincutPool
        self.pool_mlp = nn.Linear(self.gnn_hidden, self.num_clusters)

        # Componente 1: Proyección a espacio latente
        pooled_dim = self.gnn_hidden * self.num_clusters
        self.fc_mu = nn.Linear(pooled_dim, self.latent_dim)
        self.fc_log_std = nn.Linear(pooled_dim, self.latent_dim)

        # Componente 2: Decoder para VAE (reconstrucción de adyacencia)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_clusters * self.num_clusters)
        )

        # Componente 3: Procesamiento Temporal (LSTM)
        # La entrada es determinista: [mu, log_std]
        lstm_input_dim = self.latent_dim * 2
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Componente 4: Clasificador de Salida
        self.classifier = nn.Linear(lstm_hidden * 2, 1)

    def reparameterize(self, mu, log_std):
        """Reparameterization trick: z = μ + σ * ε"""
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batched_graphs_by_layer):
        """
        Procesa una secuencia de batches de grafos.
        
        Args:
            batched_graphs_by_layer: Lista de objetos `Batch` de PyG, uno por cada capa/tiempo.
        
        Returns:
            logits (Tensor): Logits crudos para clasificación.
            total_aux_loss (Tensor): Suma de todas las pérdidas auxiliares (VAE + Mincut).
        """
        lstm_inputs = []
        total_aux_loss = 0.0
        
        # Procesar cada grafo de la secuencia temporal
        for layer_data in batched_graphs_by_layer:
            x, edge_index, edge_attr, batch = layer_data.x, layer_data.edge_index, layer_data.edge_attr, layer_data.batch

            # 1. Encoder GINE
            x_gnn = self.conv1(x, edge_index, edge_attr)
            x_gnn = self.bn1(x_gnn)
            x_gnn = F.relu(x_gnn)
            
            x_gnn = self.conv2(x_gnn, edge_index, edge_attr)
            x_gnn = self.bn2(x_gnn)

            # 2. Conversión a formato Denso para MincutPool
            # Shape x_dense: [batch_size, max_nodes, gnn_hidden]
            # Shape mask: [batch_size, max_nodes]
            x_dense, mask = to_dense_batch(x_gnn, batch, max_num_nodes=200)
            
            # Shape adj: [batch_size, max_nodes, max_nodes]
            adj = to_dense_adj(edge_index, batch, max_num_nodes=200)

            # 3. Pooling Jerárquico (dense_mincut_pool)
            # Aprender la matriz de asignación de clusters
            s = self.pool_mlp(x_dense) # Shape s: [batch_size, max_nodes, num_clusters]

            # Aplicar pooling
            # Shape x_pooled: [batch_size, num_clusters, gnn_hidden]
            # Shape adj_pooled: [batch_size, num_clusters, num_clusters]
            x_pooled, adj_pooled, mc_loss, o_loss = dense_mincut_pool(x_dense, adj, s, mask)

            # 4. Proyección a espacio latente
            # Aplanar la salida del pooling para obtener un único vector por grafo
            # Shape graph_repr: [batch_size, num_clusters * gnn_hidden]
            graph_repr = x_pooled.flatten(start_dim=1)
            
            mu = self.fc_mu(graph_repr)
            log_std = self.fc_log_std(graph_repr)

            # 5. Calcular pérdidas auxiliares del VAE
            z = self.reparameterize(mu, log_std)
            
            # Reconstrucción de la matriz de adyacencia agrupada
            adj_recons_flat = self.decoder(z)
            adj_recons = adj_recons_flat.view(-1, self.num_clusters, self.num_clusters)
            recon_loss = F.mse_loss(adj_recons, adj_pooled, reduction='mean')

            # Divergencia KL (analítica)
            kl_loss = 0.5 * torch.mean(torch.exp(2 * log_std) + mu.pow(2) - 1. - (2 * log_std))
            
            # Sumar todas las pérdidas auxiliares de esta capa
            total_aux_loss += mc_loss + o_loss + kl_loss + recon_loss

            # 6. Preparar entrada para el LSTM (determinista)
            # Shape lstm_input: [batch_size, latent_dim * 2]
            lstm_input = torch.cat([mu, log_std], dim=-1)
            lstm_inputs.append(lstm_input)

        # 7. Procesamiento Temporal con LSTM
        # Shape lstm_sequence: [batch_size, num_layers, latent_dim * 2]
        lstm_sequence = torch.stack(lstm_inputs, dim=1)
        
        lstm_out, (h_n, c_n) = self.lstm(lstm_sequence)
        
        # Usar el último estado oculto (concatenado de ambas direcciones)
        # Shape final_hidden: [batch_size, lstm_hidden * 2]
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)

        # 8. Clasificación final para obtener logits
        logits = self.classifier(final_hidden) # Shape: [batch_size, 1]
        
        return logits, total_aux_loss / len(batched_graphs_by_layer)


# ============================================================================
# DATASET, TRAINING, Y EVALUACIÓN (Adaptado para la nueva arquitectura)
# ============================================================================

class PreprocessedGNNDataset(IterableDataset):
    def __init__(self, preprocessed_dir, batch_files_to_load=None, shuffle_buffer_size=500):
        super().__init__()
        self.preprocessed_dir = Path(preprocessed_dir)
        self.shuffle_buffer_size = shuffle_buffer_size
        all_files = sorted(list(self.preprocessed_dir.glob('preprocessed_*.pt')))
        if not all_files: raise ValueError(f"No se encontraron archivos en {preprocessed_dir}")
        self.batch_files = all_files if batch_files_to_load is None else [Path(f) for f in batch_files_to_load]
        
        print(f"📦 Escaneando {len(self.batch_files)} archivos batch...")
        total_traces = sum(len(torch.load(f, map_location='cpu')['graphs']) for f in tqdm(self.batch_files, desc="Escaneando"))
        self.total_traces = total_traces
        print(f"✅ Dataset GNN (Iterable): {total_traces} traces en {len(self.batch_files)} archivos")

    def __len__(self): return self.total_traces
    def _get_worker_files(self):
        worker_info = torch.utils.data.get_worker_info()
        return self.batch_files if worker_info is None else [f for i, f in enumerate(self.batch_files) if i % worker_info.num_workers == worker_info.id]
    def _generate_samples(self):
        for batch_file in self._get_worker_files():
            batch_data = torch.load(batch_file, map_location='cpu')
            for i in range(len(batch_data['graphs'])):
                yield (batch_data['graphs'][i], batch_data['labels'][i], batch_data['question_ids'][i])
            del batch_data; gc.collect()
    def __iter__(self):
        return iter(self._generate_samples())

def collate_gnn_batch(batch):
    if not batch: return None, None, None
    num_layers = len(batch[0][0])
    batched_by_layer = [PyGBatch.from_data_list([item[0][l] for item in batch]) for l in range(num_layers)]
    labels = torch.stack([item[1] for item in batch])
    question_ids = [item[2] for item in batch]
    return batched_by_layer, labels, question_ids

def find_optimal_threshold(labels, probs):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, probs)
    if len(thresholds) == 0: return 0.5
    return thresholds[np.argmax(tpr - fpr)]

def evaluate_model(model, data_loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_data in data_loader:
            batched_by_layer, labels, _ = batch_data
            if batched_by_layer is None: continue
            
            batched_by_layer_gpu = [d.to(device, non_blocking=True) for d in batched_by_layer]
            logits, _ = model(batched_by_layer_gpu)
            
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    if len(all_labels) == 0: return {m: 0 for m in ['auroc', 'accuracy', 'precision', 'recall', 'f1']}
    
    preds = (all_probs > threshold).astype(float)
    return {'auroc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
            'accuracy': accuracy_score(all_labels, preds),
            'precision': precision_score(all_labels, preds, zero_division=0),
            'recall': recall_score(all_labels, preds, zero_division=0),
            'f1': f1_score(all_labels, preds, zero_division=0)}

def train_model(model, optimizer, train_loader, val_loader, device, args, output_dir, start_epoch, best_val_auroc, best_threshold, history):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs_no_improve = 0
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, total_task_loss, total_aux_loss = 0, 0, 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (batched_by_layer, labels, _) in enumerate(progress_bar):
            if batched_by_layer is None: continue
            labels = labels.to(device).unsqueeze(1)
            batched_by_layer_gpu = [d.to(device) for d in batched_by_layer]
            
            optimizer.zero_grad()
            logits, aux_loss = model(batched_by_layer_gpu)
            
            task_loss = criterion(logits, labels)
            loss = task_loss + args.aux_loss_weight * aux_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_aux_loss += aux_loss.item()
            
            progress_bar.set_postfix({
                'loss': total_loss / (i + 1),
                'task': total_task_loss / (i + 1),
                'aux': total_aux_loss / (i + 1)
            })
            
            del loss, task_loss, aux_loss, logits, labels; gc.collect(); torch.cuda.empty_cache()

        # Validation
        val_probs, val_labels = [], []
        model.eval()
        with torch.no_grad():
            for batched_by_layer, labels, _ in val_loader:
                if batched_by_layer is None: continue
                batched_by_layer_gpu = [d.to(device) for d in batched_by_layer]
                logits, _ = model(batched_by_layer_gpu)
                val_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_labels.extend(labels.numpy().flatten())
        
        optimal_threshold = find_optimal_threshold(val_labels, val_probs)
        val_metrics = evaluate_model(model, val_loader, device, threshold=optimal_threshold)
        
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_task_loss = total_task_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_aux_loss = total_aux_loss / len(train_loader) if len(train_loader) > 0 else 0

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} (Task={avg_task_loss:.4f}, Aux={avg_aux_loss:.4f}), "
              f"Val AUROC={val_metrics['auroc']:.4f}, Val F1={val_metrics['f1']:.4f} (thr={optimal_threshold:.3f})")
        
        history[epoch] = {
            'train_loss': avg_train_loss,
            'train_task_loss': avg_task_loss,
            'train_aux_loss': avg_aux_loss,
            'val_metrics': val_metrics,
            'optimal_threshold': optimal_threshold
        }

        # Guardado del mejor modelo
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_threshold = optimal_threshold
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"  -> New best model saved with AUROC: {best_val_auroc:.4f}")
        else:
            epochs_no_improve += 1

        # Checkpoint de la época actual
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_auroc': best_val_auroc,
            'best_threshold': best_threshold,
            'history': history
        }
        torch.save(checkpoint, output_dir / 'latest_checkpoint.pt')

        # Early Stopping
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs with no improvement.")
            break
            
    return history, best_threshold

def run_experiment(args):
    print("="*80 + "\nEntrenamiento del Modelo GraphSequenceClassifier\n" + "="*80)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Dispositivo: {device}")
    if device.type == 'cuda': torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Resultados se guardarán en: {output_dir.resolve()}")

    # --- Carga de Datos ---
    gnn_dir = Path(args.preprocessed_dir) / 'gnn'
    all_files = sorted(list(gnn_dir.glob('preprocessed_*.pt')))
    random.seed(42); random.shuffle(all_files)
    train_idx, val_idx = int(0.7 * len(all_files)), int(0.85 * len(all_files))
    train_files, val_files, test_files = all_files[:train_idx], all_files[train_idx:val_idx], all_files[val_idx:]
    
    num_workers = min(len(train_files), multiprocessing.cpu_count(), 4)
    train_loader = DataLoader(PreprocessedGNNDataset(gnn_dir, train_files), batch_size=args.batch_size, collate_fn=collate_gnn_batch, num_workers=num_workers)
    val_loader = DataLoader(PreprocessedGNNDataset(gnn_dir, val_files), batch_size=args.batch_size, collate_fn=collate_gnn_batch, num_workers=num_workers)
    test_loader = DataLoader(PreprocessedGNNDataset(gnn_dir, test_files), batch_size=args.batch_size, collate_fn=collate_gnn_batch, num_workers=num_workers)

    hidden_dim = next(iter(train_loader))[0][0].x.shape[-1]
    print(f"Dimensión de hidden states: {hidden_dim}")

    # --- Inicialización del Modelo y Optimizador ---
    model = GraphSequenceClassifier(hidden_dim=hidden_dim, gnn_hidden=args.gnn_hidden, latent_dim=args.latent_dim,
                                    lstm_hidden=args.lstm_hidden, num_lstm_layers=args.num_lstm_layers, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # --- Lógica para Reanudar Entrenamiento ---
    start_epoch = 0
    best_val_auroc = 0.0
    best_threshold = 0.5
    history = {}
    checkpoint_path = output_dir / 'latest_checkpoint.pt'

    if args.resume and checkpoint_path.exists():
        print(f"Reanudando entrenamiento desde {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_auroc = checkpoint['best_val_auroc']
        best_threshold = checkpoint.get('best_threshold', 0.5)
        history = checkpoint['history']
        model.to(device)
        print(f"Reanudado desde la época {start_epoch}. Mejor Val AUROC hasta ahora: {best_val_auroc:.4f}")
    else:
        print("Iniciando nueva sesión de entrenamiento.")

    # --- Entrenamiento ---
    history, final_best_threshold = train_model(
        model, optimizer, train_loader, val_loader, device, args,
        output_dir, start_epoch, best_val_auroc, best_threshold, history
    )
    
    # --- Evaluación Final en Test ---
    print("\n" + "="*80 + "\nEVALUACIÓN FINAL EN TEST\n" + "="*80)
    best_model_path = output_dir / 'best_model.pt'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        test_metrics = evaluate_model(model, test_loader, device, threshold=final_best_threshold)
        
        print(f"Métricas en TEST (threshold={final_best_threshold:.3f}):")
        for metric, value in test_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        history['final_test_metrics'] = test_metrics
        history['final_best_threshold'] = final_best_threshold
    else:
        print("No se encontró el mejor modelo ('best_model.pt'). Saltando evaluación final.")
        history['final_test_metrics'] = "Skipped"

    # --- Guardado de Resultados ---
    results_file = output_dir / f'GraphSequenceClassifier_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    # Convertir tensores y otros objetos no serializables
    serializable_history = json.loads(json.dumps(history, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else str(o)))
    
    with open(results_file, 'w') as f:
        json.dump(serializable_history, f, indent=4)
    print(f"\nResultados finales guardados en: {results_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenar GraphSequenceClassifier.")
    parser.add_argument('--preprocessed-dir', type=str, required=True, help='Directorio con datos preprocesados ("gnn")')
    parser.add_argument('--output-dir', type=str, default='dyngad_results', help='Directorio para guardar checkpoints y resultados')
    parser.add_argument('--gnn-hidden', type=int, default=128)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--lstm-hidden', type=int, default=128)
    parser.add_argument('--num-lstm-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (reducir si hay problemas de memoria)')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--aux-loss-weight', type=float, default=1.0, help='Peso para la pérdida auxiliar combinada')
    parser.add_argument('--patience', type=int, default=20, help='Paciencia para Early Stopping')
    parser.add_argument('--resume', action='store_true', help='Indica si se debe reanudar el entrenamiento desde el último checkpoint')
    parser.add_argument('--force-cpu', action='store_true')
    
    args = parser.parse_args()
    run_experiment(args)
