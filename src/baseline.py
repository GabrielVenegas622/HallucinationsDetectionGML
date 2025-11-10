"""
Experimentos de Ablación: Comparación de 3 Arquitecturas

Este script implementa la estrategia de ablación para probar la hipótesis:
"La dinámica estructural secuencial a través de las capas es la señal clave"

MODELOS:
1. LSTM-solo (Baseline): Sin estructura de grafo, solo secuencia de capas
2. GNN-det+LSTM (CHARM-style): Con estructura de grafo (determinista)
3. GVAE+LSTM (Propuesto): Con estructura + modelado de incertidumbre (variacional)

HIPÓTESIS A PROBAR:
Si GVAE+LSTM > GNN-det+LSTM > LSTM-solo, entonces la estructura del grafo
Y el modelado de incertidumbre aportan valor incremental.

METODOLOGÍA HaloScope:
- Aplicar threshold sobre scores BLEURT para obtener etiquetas binarias (0/1)
- Usar Binary Cross Entropy como función de pérdida
- AUROC como métrica principal de evaluación

Uso:
    python baseline.py --data-pattern "traces_data/*.pkl" --scores-file ground_truth_scores.csv --epochs 50 --score-threshold 0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool
from torch_geometric.loader import DataLoader as PyGDataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score

from dataloader import TraceGraphDataset


# ============================================================================
# MODELO 1: LSTM-solo (Baseline sin estructura)
# ============================================================================

class LSTMBaseline(nn.Module):
    """
    Baseline: Solo LSTM sobre la secuencia de capas.
    
    Ignora completamente la estructura del grafo. Cada capa se representa
    por la media de sus hidden states, creando una secuencia temporal.
    
    Input: Secuencia de num_layers vectores (cada uno de dim hidden_dim)
    Output: Score de hallucination
    """
    def __init__(self, hidden_dim, lstm_hidden=256, num_lstm_layers=2, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        
        # LSTM bidireccional para capturar dependencias en ambas direcciones
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Capas de clasificación
        self.fc1 = nn.Linear(lstm_hidden * 2, 128)  # *2 por bidireccional
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, layer_sequence):
        """
        Args:
            layer_sequence: Tensor [batch_size, num_layers, hidden_dim]
        Returns:
            logits: Tensor [batch_size, 1] - logits para clasificación binaria
        """
        # LSTM sobre la secuencia de capas
        lstm_out, (h_n, c_n) = self.lstm(layer_sequence)
        
        # Usar el último estado oculto (concatenado de ambas direcciones)
        # h_n shape: [num_layers*2, batch_size, lstm_hidden]
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch_size, lstm_hidden*2]
        
        # Clasificación binaria
        x = F.relu(self.fc1(final_hidden))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)  # [batch_size, 1] - sin sigmoid, se aplica en la loss
        
        return logits


# ============================================================================
# MODELO 2: GNN-det+LSTM (CHARM-style con estructura determinista)
# ============================================================================

class GNNDetLSTM(nn.Module):
    """
    GNN determinista + LSTM: Procesa la estructura del grafo en cada capa,
    luego usa LSTM para capturar la dinámica temporal entre capas.
    
    IMPORTANTE: Usa edge_attr (pesos de atención) en las capas GNN.
    
    Similar a CHARM pero simplificado para ablación limpia.
    
    Pipeline:
    1. Para cada capa: GINE (con edge features) extrae representación considerando estructura Y pesos
    2. Secuencia de representaciones por capa → LSTM
    3. Clasificación final
    """
    def __init__(self, hidden_dim, gnn_hidden=128, lstm_hidden=256, 
                 num_lstm_layers=2, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.gnn_hidden = gnn_hidden
        
        # GINE para procesar estructura del grafo CON edge features (pesos de atención)
        # GINEConv: Graph Isomorphism Network with Edge Features
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, gnn_hidden),
                nn.ReLU(),
                nn.Linear(gnn_hidden, gnn_hidden)
            ),
            edge_dim=1  # edge_attr es 1-dimensional (valor de atención)
        )
        
        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(gnn_hidden, gnn_hidden),
                nn.ReLU(),
                nn.Linear(gnn_hidden, gnn_hidden)
            ),
            edge_dim=1
        )
        
        # LSTM sobre la secuencia de representaciones de grafos
        self.lstm = nn.LSTM(
            input_size=gnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Clasificación
        self.fc1 = nn.Linear(lstm_hidden * 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, batched_graphs_by_layer, num_layers):
        """
        Args:
            batched_graphs_by_layer: Lista de PyG Data objects, uno por capa
            num_layers: Número de capas
        Returns:
            logits: Tensor [batch_size, 1] - logits para clasificación binaria
        """
        layer_representations = []
        
        # Procesar cada capa con GINE (usando edge_attr)
        for layer_idx, layer_data in enumerate(batched_graphs_by_layer):
            x, edge_index, edge_attr, batch = (
                layer_data.x, 
                layer_data.edge_index, 
                layer_data.edge_attr,
                layer_data.batch
            )
            
            # Validar que x no tenga NaN o Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"WARNING: NaN o Inf detectado en x de capa {layer_idx}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Manejo seguro de edge_attr
            if edge_attr is not None and edge_attr.numel() > 0:
                # Validar que edge_attr no tenga NaN o Inf
                if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                    print(f"WARNING: NaN o Inf detectado en edge_attr de capa {layer_idx}")
                    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Asegurar que edge_attr tenga la forma correcta [num_edges, 1]
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(1)
                
                # Verificar que el número de arcos coincide
                if edge_attr.size(0) != edge_index.size(1):
                    print(f"WARNING: Mismatch en capa {layer_idx}: edge_attr={edge_attr.size(0)}, edges={edge_index.size(1)}")
                    # Ajustar edge_attr al tamaño correcto
                    num_edges = edge_index.size(1)
                    if edge_attr.size(0) > num_edges:
                        edge_attr = edge_attr[:num_edges]
                    else:
                        padding = torch.zeros((num_edges - edge_attr.size(0), 1), 
                                            dtype=edge_attr.dtype, device=edge_attr.device)
                        edge_attr = torch.cat([edge_attr, padding], dim=0)
                
                # Clip valores extremos en edge_attr
                edge_attr = torch.clamp(edge_attr, min=0.0, max=1.0)
            else:
                # Si no hay arcos, crear edge_attr vacío con forma correcta
                edge_attr = torch.zeros((0, 1), dtype=torch.float, device=x.device)
            
            # GINE encoding: propaga información considerando pesos de atención
            try:
                x = F.relu(self.conv1(x, edge_index, edge_attr))
                x = F.dropout(x, p=0.2, training=self.training)
                x = self.conv2(x, edge_index, edge_attr)
            except RuntimeError as e:
                print(f"ERROR en GINE de capa {layer_idx}:")
                print(f"  x.shape: {x.shape}, device: {x.device}, dtype: {x.dtype}")
                print(f"  edge_index.shape: {edge_index.shape}, num_edges: {edge_index.size(1)}")
                print(f"  edge_attr.shape: {edge_attr.shape}")
                print(f"  Rango de x: [{x.min().item():.4f}, {x.max().item():.4f}]")
                print(f"  Rango de edge_attr: [{edge_attr.min().item():.4f}, {edge_attr.max().item():.4f}]")
                raise e
            
            # Validar salida
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"WARNING: NaN o Inf en salida de GINE capa {layer_idx}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Global pooling: un vector por grafo en el batch
            graph_repr = global_mean_pool(x, batch)  # [batch_size, gnn_hidden]
            layer_representations.append(graph_repr)
        
        # Stack para crear secuencia temporal
        # [batch_size, num_layers, gnn_hidden]
        layer_sequence = torch.stack(layer_representations, dim=1)
        
        # LSTM sobre la secuencia
        lstm_out, (h_n, c_n) = self.lstm(layer_sequence)
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # Clasificación binaria
        x = F.relu(self.fc1(final_hidden))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)  # [batch_size, 1] - sin sigmoid, se aplica en la loss
        
        return logits


# ============================================================================
# MODELO 3: GVAE+LSTM (Propuesto: Estructura + Incertidumbre Variacional)
# ============================================================================

class GVAELSTM(nn.Module):
    """
    Graph Variational Autoencoder + LSTM: Modela la incertidumbre en la
    representación del grafo usando un enfoque variacional.
    
    IMPORTANTE: Usa edge_attr (pesos de atención) en las capas GNN.
    
    Pipeline:
    1. Para cada capa: GVAE encoder (con edge features) → distribución latente z ~ N(μ, σ²)
    2. Sampling de z (reparameterization trick)
    3. Secuencia de z's → LSTM
    4. Clasificación + pérdida de reconstrucción (regularización)
    
    La incertidumbre capturada puede ayudar a detectar alucinaciones.
    """
    def __init__(self, hidden_dim, gnn_hidden=128, latent_dim=64, 
                 lstm_hidden=256, num_lstm_layers=2, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.gnn_hidden = gnn_hidden
        self.latent_dim = latent_dim
        
        # GVAE Encoder: GINE (con edge features) + proyección a distribución
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, gnn_hidden),
                nn.ReLU(),
                nn.Linear(gnn_hidden, gnn_hidden)
            ),
            edge_dim=1  # edge_attr es 1-dimensional (valor de atención)
        )
        
        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(gnn_hidden, gnn_hidden),
                nn.ReLU(),
                nn.Linear(gnn_hidden, gnn_hidden)
            ),
            edge_dim=1
        )
        
        # Proyecciones a parámetros de la distribución latente
        self.fc_mu = nn.Linear(gnn_hidden, latent_dim)
        self.fc_logvar = nn.Linear(gnn_hidden, latent_dim)
        
        # GVAE Decoder (para regularización)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, gnn_hidden),
            nn.ReLU(),
            nn.Linear(gnn_hidden, gnn_hidden)
        )
        
        # LSTM sobre secuencia de representaciones latentes
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Clasificación
        self.fc1 = nn.Linear(lstm_hidden * 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def encode(self, x, edge_index, edge_attr, batch):
        """Encoder: grafo (con edge features) → distribución latente"""
        
        # Validar que x no tenga NaN o Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"WARNING: NaN o Inf detectado en x del encoder")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Manejo seguro de edge_attr
        if edge_attr is not None and edge_attr.numel() > 0:
            # Validar que edge_attr no tenga NaN o Inf
            if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                print(f"WARNING: NaN o Inf detectado en edge_attr del encoder")
                edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Asegurar que edge_attr tenga la forma correcta [num_edges, 1]
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            
            # Verificar que el número de arcos coincide
            if edge_attr.size(0) != edge_index.size(1):
                num_edges = edge_index.size(1)
                if edge_attr.size(0) > num_edges:
                    edge_attr = edge_attr[:num_edges]
                else:
                    padding = torch.zeros((num_edges - edge_attr.size(0), 1), 
                                        dtype=edge_attr.dtype, device=edge_attr.device)
                    edge_attr = torch.cat([edge_attr, padding], dim=0)
            
            # Clip valores extremos
            edge_attr = torch.clamp(edge_attr, min=0.0, max=1.0)
        else:
            # Si no hay arcos, crear edge_attr vacío con forma correcta
            edge_attr = torch.zeros((0, 1), dtype=torch.float, device=x.device)
        
        # GINE: usa edge features (pesos de atención)
        try:
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv2(x, edge_index, edge_attr)
        except RuntimeError as e:
            print(f"ERROR en GINE encoder:")
            print(f"  x.shape: {x.shape}, device: {x.device}")
            print(f"  edge_index.shape: {edge_index.shape}")
            print(f"  edge_attr.shape: {edge_attr.shape}")
            raise e
        
        # Validar salida
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"WARNING: NaN o Inf en salida de encoder GINE")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Global pooling
        graph_repr = global_mean_pool(x, batch)
        
        # Parámetros de la distribución
        mu = self.fc_mu(graph_repr)
        logvar = self.fc_logvar(graph_repr)
        
        # Clip logvar para evitar valores extremos
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mu, logvar, graph_repr
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ * ε, donde ε ~ N(0,1)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decoder: representación latente → reconstrucción"""
        return self.decoder(z)
    
    def forward(self, batched_graphs_by_layer, num_layers):
        """
        Args:
            batched_graphs_by_layer: Lista de PyG Data objects
            num_layers: Número de capas
        Returns:
            logits: Tensor [batch_size, 1] - logits para clasificación binaria
            mu_list: Lista de tensores mu (para KL loss)
            logvar_list: Lista de tensores logvar (para KL loss)
            original_reprs: Lista de representaciones originales (para reconstruction loss)
            reconstructed_reprs: Lista de reconstrucciones
        """
        latent_sequence = []
        mu_list = []
        logvar_list = []
        original_reprs = []
        reconstructed_reprs = []
        
        # Procesar cada capa con GVAE (usando edge_attr)
        for layer_data in batched_graphs_by_layer:
            x, edge_index, edge_attr, batch = (
                layer_data.x, 
                layer_data.edge_index, 
                layer_data.edge_attr,  # ← AHORA USAMOS ESTO
                layer_data.batch
            )
            
            # Encode (con edge features)
            mu, logvar, graph_repr = self.encode(x, edge_index, edge_attr, batch)
            
            # Reparameterize
            z = self.reparameterize(mu, logvar)
            
            # Decode (para pérdida de reconstrucción)
            x_reconstructed = self.decode(z)
            
            # Guardar para pérdidas
            mu_list.append(mu)
            logvar_list.append(logvar)
            original_reprs.append(graph_repr)
            reconstructed_reprs.append(x_reconstructed)
            
            latent_sequence.append(z)
        
        # Secuencia latente
        latent_seq = torch.stack(latent_sequence, dim=1)  # [batch_size, num_layers, latent_dim]
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(latent_seq)
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # Clasificación binaria
        x = F.relu(self.fc1(final_hidden))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)  # [batch_size, 1] - sin sigmoid, se aplica en la loss
        
        return logits, mu_list, logvar_list, original_reprs, reconstructed_reprs


# ============================================================================
# FUNCIONES DE PÉRDIDA
# ============================================================================

def vae_loss(recon_x, x, mu, logvar, kl_weight=0.001):
    """
    Pérdida VAE = Reconstruction Loss + KL Divergence
    
    Args:
        recon_x: Reconstrucciones
        x: Originales
        mu: Medias de la distribución latente
        logvar: Log-varianzas de la distribución latente
        kl_weight: Peso para la divergencia KL
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_weight * kld


# ============================================================================
# DATASET Y LOADERS PERSONALIZADOS
# ============================================================================

class SequentialTraceDataset:
    """
    Dataset que organiza grafos por trace completo (todas las capas juntas)
    en lugar de grafos individuales.
    
    Aplica threshold sobre scores BLEURT para obtener etiquetas binarias:
    - label = 1 si score < threshold (alucinación)
    - label = 0 si score >= threshold (no alucinación)
    """
    def __init__(self, pkl_files_pattern, scores_file, attn_threshold=0.01, score_threshold=0.5):
        from dataloader import TraceGraphDataset
        
        # Cargar dataset de grafos
        self.graph_dataset = TraceGraphDataset(pkl_files_pattern, attn_threshold)
        
        # Cargar scores y convertir a etiquetas binarias
        scores_df = pd.read_csv(scores_file)
        self.scores_dict = dict(zip(scores_df['question_id'], scores_df['bleurt_score']))
        
        # Threshold para convertir scores a etiquetas binarias
        self.score_threshold = score_threshold
        
        # Crear diccionario de etiquetas binarias
        self.labels_dict = {}
        for qid, score in self.scores_dict.items():
            # label = 1 si es alucinación (score bajo), 0 si no
            self.labels_dict[qid] = 1 if score < score_threshold else 0
        
        self.num_layers = self.graph_dataset.num_layers
        self.num_traces = len(self.graph_dataset.all_traces)
        
        # Calcular estadísticas de balance de clases
        labels_values = list(self.labels_dict.values())
        num_hallucinations = sum(labels_values)
        num_correct = len(labels_values) - num_hallucinations
        
        print(f"Dataset secuencial creado:")
        print(f"  - {self.num_traces} traces")
        print(f"  - {self.num_layers} capas por trace")
        print(f"  - {len(self.scores_dict)} scores cargados")
        print(f"  - Score threshold: {score_threshold}")
        print(f"  - Balance de clases:")
        print(f"    * Alucinaciones (1): {num_hallucinations} ({100*num_hallucinations/len(labels_values):.1f}%)")
        print(f"    * No alucinaciones (0): {num_correct} ({100*num_correct/len(labels_values):.1f}%)")
    
    def __len__(self):
        return self.num_traces
    
    def __getitem__(self, trace_idx):
        """
        Retorna todos los grafos de un trace (una por capa) + etiqueta binaria.
        
        Returns:
            graphs_by_layer: Lista de Data objects
            label: Etiqueta binaria (0 o 1)
            question_id: ID de la pregunta
        """
        graphs_by_layer = []
        question_id = None
        
        for layer_idx in range(self.num_layers):
            # Calcular índice global en el dataset plano
            global_idx = trace_idx * self.num_layers + layer_idx
            graph = self.graph_dataset[global_idx]
            graphs_by_layer.append(graph)
            
            if question_id is None:
                question_id = graph.question_id
        
        # Obtener etiqueta binaria
        label = self.labels_dict.get(question_id, 0)
        
        return graphs_by_layer, torch.tensor(label, dtype=torch.float), question_id


def collate_sequential_batch(batch):
    """
    Collate function para organizar batches de secuencias de grafos.
    
    Args:
        batch: Lista de (graphs_by_layer, label, question_id)
    
    Returns:
        batched_by_layer: Lista de num_layers batches de PyG
        labels: Tensor de etiquetas binarias
        question_ids: Lista de IDs
    """
    from torch_geometric.data import Batch as PyGBatch
    
    if len(batch) == 0:
        raise ValueError("Batch vacío recibido en collate_fn")
    
    batch_size = len(batch)
    num_layers = len(batch[0][0])
    
    # Reorganizar: en lugar de [batch][layer], queremos [layer][batch]
    batched_by_layer = []
    for layer_idx in range(num_layers):
        try:
            layer_graphs = [item[0][layer_idx] for item in batch]
            
            # Validar que todos los grafos tengan edge_attr
            for i, graph in enumerate(layer_graphs):
                if graph.edge_attr is None or graph.edge_attr.numel() == 0:
                    # Si no hay edge_attr, crear uno vacío con forma correcta
                    graph.edge_attr = torch.zeros((0, 1), dtype=torch.float)
                elif graph.edge_attr.dim() == 1:
                    # Asegurar que sea 2D
                    graph.edge_attr = graph.edge_attr.unsqueeze(1)
                
                # Verificar consistencia entre edge_index y edge_attr
                num_edges = graph.edge_index.size(1)
                if graph.edge_attr.size(0) != num_edges:
                    # Crear edge_attr del tamaño correcto si hay mismatch
                    print(f"WARNING: Corrigiendo edge_attr en capa {layer_idx}, grafo {i}: "
                          f"{graph.edge_attr.size(0)} != {num_edges}")
                    if num_edges > 0:
                        # Tomar los primeros num_edges o rellenar con zeros
                        if graph.edge_attr.size(0) > num_edges:
                            graph.edge_attr = graph.edge_attr[:num_edges]
                        else:
                            # Rellenar con valores promedio
                            padding = torch.zeros((num_edges - graph.edge_attr.size(0), 1), 
                                                 dtype=torch.float)
                            graph.edge_attr = torch.cat([graph.edge_attr, padding], dim=0)
                    else:
                        graph.edge_attr = torch.zeros((0, 1), dtype=torch.float)
            
            batched_layer = PyGBatch.from_data_list(layer_graphs)
            batched_by_layer.append(batched_layer)
        except Exception as e:
            print(f"Error al procesar capa {layer_idx}: {e}")
            print(f"Número de grafos en batch: {len(layer_graphs)}")
            for i, graph in enumerate(layer_graphs):
                print(f"  Grafo {i}: nodes={graph.num_nodes}, edges={graph.edge_index.size(1)}, "
                      f"edge_attr={'None' if graph.edge_attr is None else graph.edge_attr.shape}")
            raise
    
    labels = torch.stack([item[1] for item in batch])
    question_ids = [item[2] for item in batch]
    
    return batched_by_layer, labels, question_ids


# ============================================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================================

def train_lstm_baseline(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    """Entrena el modelo LSTM Baseline con clasificación binaria"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()  # BCE con logits (más estable numéricamente)
    
    history = {'train_loss': [], 'val_loss': [], 'val_auroc': [], 'val_acc': [], 'val_f1': []}
    best_val_auroc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batched_by_layer, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            labels = labels.to(device).unsqueeze(1)
            
            # Extraer secuencia: promediar nodos de cada grafo en cada capa
            layer_sequence = []
            for layer_data in batched_by_layer:
                # Promedio global de nodos por grafo
                from torch_geometric.nn import global_mean_pool
                layer_repr = global_mean_pool(layer_data.x.to(device), 
                                              layer_data.batch.to(device))
                layer_sequence.append(layer_repr)
            
            layer_sequence = torch.stack(layer_sequence, dim=1)  # [batch, layers, dim]
            
            optimizer.zero_grad()
            logits = model(layer_sequence)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_probs = []
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batched_by_layer, labels, _ in val_loader:
                labels = labels.to(device).unsqueeze(1)
                
                layer_sequence = []
                for layer_data in batched_by_layer:
                    from torch_geometric.nn import global_mean_pool
                    layer_repr = global_mean_pool(layer_data.x.to(device),
                                                  layer_data.batch.to(device))
                    layer_sequence.append(layer_repr)
                
                layer_sequence = torch.stack(layer_sequence, dim=1)
                logits = model(layer_sequence)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Calcular probabilidades y predicciones
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Calcular métricas
        val_auroc = roc_auc_score(all_labels, all_probs)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"AUROC={val_auroc:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
        
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), 'best_lstm_baseline.pt')
    
    return history


def train_gnn_det_lstm(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    """Entrena el modelo GNN-det+LSTM con clasificación binaria"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_auroc': [], 'val_acc': [], 'val_f1': []}
    best_val_auroc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batched_by_layer, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            labels = labels.to(device).unsqueeze(1)
            
            # Mover datos a device
            batched_by_layer = [data.to(device) for data in batched_by_layer]
            
            optimizer.zero_grad()
            logits = model(batched_by_layer, len(batched_by_layer))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        all_probs = []
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batched_by_layer, labels, _ in val_loader:
                labels = labels.to(device).unsqueeze(1)
                batched_by_layer = [data.to(device) for data in batched_by_layer]
                
                logits = model(batched_by_layer, len(batched_by_layer))
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Calcular probabilidades y predicciones
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Calcular métricas
        val_auroc = roc_auc_score(all_labels, all_probs)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"AUROC={val_auroc:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
        
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), 'best_gnn_det_lstm.pt')
    
    return history


def train_gvae_lstm(model, train_loader, val_loader, device, epochs=50, lr=0.001, kl_weight=0.001):
    """Entrena el modelo GVAE+LSTM con clasificación binaria"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    history = {'train_loss': [], 'train_task_loss': [], 'train_vae_loss': [],
               'val_loss': [], 'val_auroc': [], 'val_acc': [], 'val_f1': []}
    best_val_auroc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_task_loss = 0
        train_vae_loss = 0
        
        for batched_by_layer, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            labels = labels.to(device).unsqueeze(1)
            batched_by_layer = [data.to(device) for data in batched_by_layer]
            
            optimizer.zero_grad()
            logits, mu_list, logvar_list, orig_list, recon_list = model(
                batched_by_layer, len(batched_by_layer)
            )
            
            # Pérdida de la tarea principal (clasificación binaria)
            task_loss = criterion(logits, labels)
            
            # Pérdida VAE acumulada sobre todas las capas
            vae_loss_total = 0
            for mu, logvar, orig, recon in zip(mu_list, logvar_list, orig_list, recon_list):
                vae_loss_total += vae_loss(recon, orig, mu, logvar, kl_weight)
            vae_loss_total /= len(mu_list)
            
            # Pérdida total
            loss = task_loss + 0.1 * vae_loss_total  # Peso reducido para VAE
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_task_loss += task_loss.item()
            train_vae_loss += vae_loss_total.item()
        
        model.eval()
        val_loss = 0
        all_probs = []
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batched_by_layer, labels, _ in val_loader:
                labels = labels.to(device).unsqueeze(1)
                batched_by_layer = [data.to(device) for data in batched_by_layer]
                
                logits, _, _, _, _ = model(batched_by_layer, len(batched_by_layer))
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Calcular probabilidades y predicciones
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
        
        train_loss /= len(train_loader)
        train_task_loss /= len(train_loader)
        train_vae_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Calcular métricas
        val_auroc = roc_auc_score(all_labels, all_probs)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        
        history['train_loss'].append(train_loss)
        history['train_task_loss'].append(train_task_loss)
        history['train_vae_loss'].append(train_vae_loss)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} (Task={train_task_loss:.4f}, VAE={train_vae_loss:.4f}), "
              f"Val Loss={val_loss:.4f}, AUROC={val_auroc:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
        
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), 'best_gvae_lstm.pt')
    
    return history


# ============================================================================
# FUNCIÓN PRINCIPAL: EXPERIMENTOS DE ABLACIÓN
# ============================================================================

def run_ablation_experiments(args):
    """
    Ejecuta los 3 experimentos de ablación y compara resultados.
    """
    print("="*80)
    print("EXPERIMENTOS DE ABLACIÓN - PRUEBA DE HIPÓTESIS (Metodología HaloScope)")
    print("="*80)
    print("\nHipótesis: La dinámica estructural secuencial a través de las capas")
    print("           es la señal clave para detectar alucinaciones.")
    print("\nModelos a comparar:")
    print("  1. LSTM-solo (Baseline sin estructura)")
    print("  2. GNN-det+LSTM (Con estructura determinista)")
    print("  3. GVAE+LSTM (Con estructura + incertidumbre variacional)")
    print("\nEsperado: GVAE+LSTM > GNN-det+LSTM > LSTM-solo")
    print("\nMétrica principal: AUROC (Area Under ROC Curve)")
    print(f"Threshold de score: {args.score_threshold} (scores < threshold = alucinación)")
    print("="*80)
    
    # Configurar device
    if args.force_cpu:
        device = torch.device('cpu')
        print(f"\n⚠️  Modo CPU forzado (puede ser más lento)")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDispositivo: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Limpiar cache de CUDA
        torch.cuda.empty_cache()
        print("Cache de CUDA limpiado")
    
    # Crear dataset
    print("\nCargando dataset...")
    full_dataset = SequentialTraceDataset(
        args.data_pattern,
        args.scores_file,
        attn_threshold=args.attn_threshold,
        score_threshold=args.score_threshold
    )
    
    # Split train/val/test
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    from torch.utils.data import random_split
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nSplit del dataset:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Crear dataloaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_sequential_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_sequential_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_sequential_batch)
    
    # Obtener dimensiones
    sample_graph = full_dataset[0][0][0]
    hidden_dim = sample_graph.x.shape[1]
    print(f"\nDimensión de hidden states: {hidden_dim}")
    
    results = {}
    
    # ========================================================================
    # EXPERIMENTO 1: LSTM Baseline
    # ========================================================================
    if args.run_lstm:
        print("\n" + "="*80)
        print("EXPERIMENTO 1: LSTM-solo (Baseline)")
        print("="*80)
        
        model_lstm = LSTMBaseline(
            hidden_dim=hidden_dim,
            lstm_hidden=args.lstm_hidden,
            num_lstm_layers=args.num_lstm_layers,
            dropout=args.dropout
        )
        
        print(f"Parámetros del modelo: {sum(p.numel() for p in model_lstm.parameters()):,}")
        
        history_lstm = train_lstm_baseline(
            model_lstm, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr
        )
        
        results['LSTM-solo'] = {
            'best_val_auroc': max(history_lstm['val_auroc']),
            'best_val_acc': max(history_lstm['val_acc']),
            'best_val_f1': max(history_lstm['val_f1']),
            'history': history_lstm
        }
    
    # ========================================================================
    # EXPERIMENTO 2: GNN-det+LSTM
    # ========================================================================
    if args.run_gnn_det:
        print("\n" + "="*80)
        print("EXPERIMENTO 2: GNN-det+LSTM (CHARM-style)")
        print("="*80)
        
        model_gnn = GNNDetLSTM(
            hidden_dim=hidden_dim,
            gnn_hidden=args.gnn_hidden,
            lstm_hidden=args.lstm_hidden,
            num_lstm_layers=args.num_lstm_layers,
            dropout=args.dropout
        )
        
        print(f"Parámetros del modelo: {sum(p.numel() for p in model_gnn.parameters()):,}")
        
        history_gnn = train_gnn_det_lstm(
            model_gnn, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr
        )
        
        results['GNN-det+LSTM'] = {
            'best_val_auroc': max(history_gnn['val_auroc']),
            'best_val_acc': max(history_gnn['val_acc']),
            'best_val_f1': max(history_gnn['val_f1']),
            'history': history_gnn
        }
    
    # ========================================================================
    # EXPERIMENTO 3: GVAE+LSTM
    # ========================================================================
    if args.run_gvae:
        print("\n" + "="*80)
        print("EXPERIMENTO 3: GVAE+LSTM (Propuesto)")
        print("="*80)
        
        model_gvae = GVAELSTM(
            hidden_dim=hidden_dim,
            gnn_hidden=args.gnn_hidden,
            latent_dim=args.latent_dim,
            lstm_hidden=args.lstm_hidden,
            num_lstm_layers=args.num_lstm_layers,
            dropout=args.dropout
        )
        
        print(f"Parámetros del modelo: {sum(p.numel() for p in model_gvae.parameters()):,}")
        
        history_gvae = train_gvae_lstm(
            model_gvae, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, kl_weight=args.kl_weight
        )
        
        results['GVAE+LSTM'] = {
            'best_val_auroc': max(history_gvae['val_auroc']),
            'best_val_acc': max(history_gvae['val_acc']),
            'best_val_f1': max(history_gvae['val_f1']),
            'history': history_gvae
        }
    
    # ========================================================================
    # RESULTADOS Y CONCLUSIONES
    # ========================================================================
    print("\n" + "="*80)
    print("RESULTADOS FINALES - TABLA DE ABLACIÓN")
    print("="*80)
    
    print("\nMétrica Principal: AUROC (Mayor es mejor)")
    print("-" * 80)
    print(f"{'Modelo':<25} {'Best AUROC':>15} {'Best Accuracy':>15} {'Best F1':>15}")
    print("-" * 80)
    
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['best_val_auroc'], reverse=True):
        print(f"{model_name:<25} {metrics['best_val_auroc']:>15.4f} "
              f"{metrics['best_val_acc']:>15.4f} {metrics['best_val_f1']:>15.4f}")
    
    print("-" * 80)
    
    # Verificar hipótesis
    print("\n" + "="*80)
    print("VERIFICACIÓN DE HIPÓTESIS")
    print("="*80)
    
    if len(results) == 3:
        lstm_auroc = results['LSTM-solo']['best_val_auroc']
        gnn_auroc = results['GNN-det+LSTM']['best_val_auroc']
        gvae_auroc = results['GVAE+LSTM']['best_val_auroc']
        
        print(f"\nLSTM-solo:     AUROC = {lstm_auroc:.4f}")
        print(f"GNN-det+LSTM:  AUROC = {gnn_auroc:.4f} ({'✓' if gnn_auroc > lstm_auroc else '✗'} mejor que LSTM-solo)")
        print(f"GVAE+LSTM:     AUROC = {gvae_auroc:.4f} ({'✓' if gvae_auroc > gnn_auroc else '✗'} mejor que GNN-det+LSTM)")
        
        if gvae_auroc > gnn_auroc > lstm_auroc:
            print("\n🎉 HIPÓTESIS CONFIRMADA:")
            print("   GVAE+LSTM > GNN-det+LSTM > LSTM-solo")
            print("   La estructura del grafo Y la incertidumbre variacional aportan valor.")
        elif gnn_auroc > lstm_auroc:
            print("\n⚠️  HIPÓTESIS PARCIALMENTE CONFIRMADA:")
            print("   La estructura del grafo aporta valor, pero la incertidumbre")
            print("   variacional no mejora significativamente.")
        else:
            print("\n❌ HIPÓTESIS NO CONFIRMADA:")
            print("   Los resultados sugieren revisar la arquitectura o hiperparámetros.")
    
    # Guardar resultados
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"ablation_results_{timestamp}.json"
    
    # Convertir historia a listas para JSON
    results_json = {}
    for model_name, metrics in results.items():
        results_json[model_name] = {
            'best_val_auroc': metrics['best_val_auroc'],
            'best_val_acc': metrics['best_val_acc'],
            'best_val_f1': metrics['best_val_f1'],
            'history': {k: v for k, v in metrics['history'].items()}
        }
    
    # Agregar configuración
    results_json['config'] = {
        'score_threshold': args.score_threshold,
        'attn_threshold': args.attn_threshold,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Resultados guardados en: {results_file}")
    print("\n" + "="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Experimentos de ablación para detección de alucinaciones (Metodología HaloScope)"
    )
    
    # Datos
    parser.add_argument('--data-pattern', type=str, required=True,
                       help='Patrón glob para archivos .pkl (ej: "traces_data/*.pkl")')
    parser.add_argument('--scores-file', type=str, required=True,
                       help='Archivo CSV con scores BLEURT')
    parser.add_argument('--attn-threshold', type=float, default=0.0,
                       help='Umbral de atención para crear arcos')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                       help='Umbral de score BLEURT para etiquetar alucinaciones (score < threshold = alucinación)')
    
    # Entrenamiento
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    # Arquitectura
    parser.add_argument('--gnn-hidden', type=int, default=128,
                       help='Dimensión oculta de GNN')
    parser.add_argument('--latent-dim', type=int, default=64,
                       help='Dimensión latente para GVAE')
    parser.add_argument('--lstm-hidden', type=int, default=256,
                       help='Dimensión oculta de LSTM')
    parser.add_argument('--num-lstm-layers', type=int, default=2,
                       help='Número de capas LSTM')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--kl-weight', type=float, default=0.001,
                       help='Peso para pérdida KL en GVAE')
    
    # Control de experimentos
    parser.add_argument('--run-lstm', action='store_true', default=True,
                       help='Ejecutar experimento LSTM-solo')
    parser.add_argument('--run-gnn-det', action='store_true', default=True,
                       help='Ejecutar experimento GNN-det+LSTM')
    parser.add_argument('--run-gvae', action='store_true', default=True,
                       help='Ejecutar experimento GVAE+LSTM')
    parser.add_argument('--output-dir', type=str, default='./ablation_results',
                       help='Directorio para guardar resultados')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Forzar ejecución en CPU (útil si hay problemas con CUDA)')
    
    args = parser.parse_args()
    
    run_ablation_experiments(args)
