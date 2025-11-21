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
    python baseline.py --data-pattern "traces_data/*.pkl*" --scores-file ground_truth_scores.csv --epochs 50 --score-threshold 0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool
from torch_geometric.loader import DataLoader as PyGDataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import gc
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from collections import deque
import random


# ============================================================================
# MODELO 1: LSTM-solo (Baseline sin estructura)
# ============================================================================

class LSTMBaseline(nn.Module):
    """
    Baseline: Solo LSTM sobre la secuencia de capas.
    
    IMPORTANTE: Ignora completamente la estructura del grafo y la información
    de atención. Solo usa el hidden state del ÚLTIMO TOKEN de cada capa,
    creando una secuencia temporal que compara cómo evoluciona el último
    token a través de las capas.
    
    Esto es clave para la ablación: NO tiene información estructural,
    solo la evolución temporal del token final.
    
    Input: Secuencia de num_layers vectores (cada uno de dim hidden_dim)
           donde cada vector es el hidden state del último token en esa capa
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
    2. Last Token Readout con conexión residual (concatena embedding original + procesado)
    3. Secuencia de representaciones por capa → LSTM
    4. Clasificación final
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
        self.bn1 = nn.BatchNorm1d(gnn_hidden)
        
        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(gnn_hidden, gnn_hidden),
                nn.ReLU(),
                nn.Linear(gnn_hidden, gnn_hidden)
            ),
            edge_dim=1
        )
        self.bn2 = nn.BatchNorm1d(gnn_hidden)
        
        # LSTM sobre la secuencia de representaciones de grafos
        # Ahora recibe hidden_dim + gnn_hidden debido a la concatenación residual
        self.lstm = nn.LSTM(
            input_size=hidden_dim + gnn_hidden,
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
        
        # Procesar cada capa con GINE (usando edge_attr) + Last Token Readout
        for layer_idx, layer_data in enumerate(batched_graphs_by_layer):
            x_original, edge_index, edge_attr, batch = (
                layer_data.x, 
                layer_data.edge_index, 
                layer_data.edge_attr,
                layer_data.batch
            )
            
            # Validar que x no tenga NaN o Inf
            if torch.isnan(x_original).any() or torch.isinf(x_original).any():
                print(f"WARNING: NaN o Inf detectado en x de capa {layer_idx}")
                x_original = torch.nan_to_num(x_original, nan=0.0, posinf=1e6, neginf=-1e6)
            
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
                edge_attr = torch.zeros((0, 1), dtype=torch.float, device=x_original.device)
            
            # GINE encoding: propaga información considerando pesos de atención
            try:
                # Capa 1: GINE -> BatchNorm -> ReLU -> Dropout
                x = self.conv1(x_original, edge_index, edge_attr)
                x = self.bn1(x)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)

                # Capa 2: GINE -> BatchNorm
                x_gnn = self.conv2(x, edge_index, edge_attr)
                x_gnn = self.bn2(x_gnn)
            except RuntimeError as e:
                print(f"ERROR en GINE de capa {layer_idx}:")
                print(f"  x.shape: {x_original.shape}, device: {x_original.device}, dtype: {x_original.dtype}")
                print(f"  edge_index.shape: {edge_index.shape}, num_edges: {edge_index.size(1)}")
                print(f"  edge_attr.shape: {edge_attr.shape}")
                print(f"  Rango de x: [{x_original.min().item():.4f}, {x_original.max().item():.4f}]")
                if edge_attr.numel() > 0:
                    print(f"  Rango de edge_attr: [{edge_attr.min().item():.4f}, {edge_attr.max().item():.4f}]")
                raise e
            
            # Validar salida
            if torch.isnan(x_gnn).any() or torch.isinf(x_gnn).any():
                print(f"WARNING: NaN o Inf en salida de GINE capa {layer_idx}")
                x_gnn = torch.nan_to_num(x_gnn, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Last Token Readout con conexión residual
            # Identificar los índices del último token de cada grafo en el batch
            batch_size = batch.max().item() + 1
            last_token_indices = []
            
            for graph_id in range(batch_size):
                # Encontrar todos los nodos que pertenecen a este grafo
                node_mask = (batch == graph_id)
                node_indices = torch.where(node_mask)[0]
                
                if len(node_indices) > 0:
                    # El último índice es el último token
                    last_token_indices.append(node_indices[-1])
                else:
                    # Caso extremo: si no hay nodos (no debería pasar), usar índice 0
                    last_token_indices.append(0)
            
            last_token_indices = torch.tensor(last_token_indices, device=x_original.device)
            
            # Extraer features del último token
            # Original: embedding antes de la GNN [batch_size, hidden_dim]
            original_last_token = x_original[last_token_indices]
            
            # GNN: embedding después de la GNN [batch_size, gnn_hidden]
            gnn_last_token = x_gnn[last_token_indices]
            
            # Concatenación residual: combinar información original + procesada
            combined = torch.cat([original_last_token, gnn_last_token], dim=1)  # [batch_size, hidden_dim + gnn_hidden]
            
            layer_representations.append(combined)
        
        # Stack para crear secuencia temporal
        # [batch_size, num_layers, hidden_dim + gnn_hidden]
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
    3. Secuencia de [original_last_token, μ, logvar] → LSTM (Two-Stage)
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
        self.bn1 = nn.BatchNorm1d(gnn_hidden)
        
        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(gnn_hidden, gnn_hidden),
                nn.ReLU(),
                nn.Linear(gnn_hidden, gnn_hidden)
            ),
            edge_dim=1
        )
        self.bn2 = nn.BatchNorm1d(gnn_hidden)
        
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
        # Two-Stage: La entrada es determinista [original_last_token, mu, logvar]
        self.lstm = nn.LSTM(
            input_size=hidden_dim + latent_dim * 2,
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
            # Capa 1: GINE -> BatchNorm -> ReLU -> Dropout
            x_conv = self.conv1(x, edge_index, edge_attr)
            x_conv = self.bn1(x_conv)
            x_conv = F.relu(x_conv)
            x_conv = F.dropout(x_conv, p=0.2, training=self.training)
            
            # Capa 2: GINE -> BatchNorm
            x_conv = self.conv2(x_conv, edge_index, edge_attr)
            x_conv = self.bn2(x_conv)

        except RuntimeError as e:
            print(f"ERROR en GINE encoder:")
            print(f"  x.shape: {x.shape}, device: {x.device}")
            print(f"  edge_index.shape: {edge_index.shape}")
            print(f"  edge_attr.shape: {edge_attr.shape}")
            raise e
        
        # Validar salida
        if torch.isnan(x_conv).any() or torch.isinf(x_conv).any():
            print(f"WARNING: NaN o Inf en salida de encoder GINE")
            x_conv = torch.nan_to_num(x_conv, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Last Token Readout: Seleccionar solo el último nodo de cada grafo
        batch_size = batch.max().item() + 1
        last_token_indices = []
        
        for graph_id in range(batch_size):
            # Encontrar todos los nodos que pertenecen a este grafo
            node_mask = (batch == graph_id)
            node_indices = torch.where(node_mask)[0]
            
            if len(node_indices) > 0:
                # El último índice es el último token
                last_token_indices.append(node_indices[-1])
            else:
                # Caso extremo: si no hay nodos (no debería pasar), usar índice 0
                last_token_indices.append(0)
        
        last_token_indices = torch.tensor(last_token_indices, device=x.device)
        
        # Extraer features del último token después de la GNN
        graph_repr = x_conv[last_token_indices]  # [batch_size, gnn_hidden]
        
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
            x_original, edge_index, edge_attr, batch = (
                layer_data.x, 
                layer_data.edge_index, 
                layer_data.edge_attr,
                layer_data.batch
            )
            
            # Extraer original_last_token ANTES de encode (sin procesar por GNN)
            batch_size = batch.max().item() + 1
            last_token_indices = []
            
            for graph_id in range(batch_size):
                # Encontrar todos los nodos que pertenecen a este grafo
                node_mask = (batch == graph_id)
                node_indices = torch.where(node_mask)[0]
                
                if len(node_indices) > 0:
                    # El último índice es el último token
                    last_token_indices.append(node_indices[-1])
                else:
                    # Caso extremo: si no hay nodos (no debería pasar), usar índice 0
                    last_token_indices.append(0)
            
            last_token_indices = torch.tensor(last_token_indices, device=x_original.device)
            
            # Extraer features originales del último token [batch_size, hidden_dim]
            original_last_token = x_original[last_token_indices]
            
            # Encode (con edge features) -> el encoder ahora usa last token readout
            mu, logvar, graph_repr = self.encode(x_original, edge_index, edge_attr, batch)
            
            # Reparameterize (SOLO para el decoder)
            z = self.reparameterize(mu, logvar)
            
            # Decode (para pérdida de reconstrucción)
            x_reconstructed = self.decode(z)
            
            # Guardar para pérdidas
            mu_list.append(mu)
            logvar_list.append(logvar)
            original_reprs.append(graph_repr)
            reconstructed_reprs.append(x_reconstructed)
            
            # Two-Stage: Concatenar info original + mu + logvar para el LSTM
            combined = torch.cat([original_last_token, mu, logvar], dim=1)
            
            latent_sequence.append(combined)
        
        # Secuencia latente
        latent_seq = torch.stack(latent_sequence, dim=1)
        
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

class PreprocessedLSTMDataset(IterableDataset):
    """
    IterableDataset para LSTM-solo con soporte para múltiples workers.
    
    ESTRATEGIA:
    - Cada worker procesa un subconjunto de archivos .pt
    - Lee un archivo completo, yield todas las traces, cierra y pasa al siguiente
    - Shuffling local mediante buffer para mejorar aleatoriedad
    - Permite paralelización real sin cargar todo en memoria
    
    VENTAJAS:
    - ✅ Soporta num_workers > 0 (paralelización real)
    - ✅ Solo 1 archivo en memoria por worker a la vez
    - ✅ Shuffling local aceptable para datasets grandes
    """
    def __init__(self, preprocessed_dir, batch_files_to_load=None, shuffle_buffer_size=1000):
        """
        Args:
            preprocessed_dir: Directorio con archivos preprocessed_*.pt
            batch_files_to_load: Lista de archivos específicos a cargar (None = cargar todos)
            shuffle_buffer_size: Tamaño del buffer para shuffling local (1000 traces ~aceptable)
        """
        super().__init__()
        self.preprocessed_dir = Path(preprocessed_dir)
        self.shuffle_buffer_size = shuffle_buffer_size
        
        # Buscar archivos de batches
        all_batch_files = sorted(list(self.preprocessed_dir.glob('preprocessed_*.pt')))
        if not all_batch_files:
            raise ValueError(f"No se encontraron archivos preprocessed_*.pt en {preprocessed_dir}")
        
        # Determinar qué archivos cargar
        if batch_files_to_load is not None:
            self.batch_files = [Path(f) for f in batch_files_to_load]
        else:
            self.batch_files = all_batch_files
        
        # Contar total de traces (solo metadata, sin cargar datos)
        print(f"📦 Escaneando {len(self.batch_files)} archivos batch (solo metadata)...")
        self.batch_sizes = []
        total_traces = 0
        
        for batch_file in tqdm(self.batch_files, desc="Escaneando"):
            batch_data = torch.load(batch_file, weights_only=False)
            batch_size = len(batch_data['sequences'])
            self.batch_sizes.append(batch_size)
            total_traces += batch_size
            del batch_data
            gc.collect()
        
        self.total_traces = total_traces
        
        print(f"✅ Dataset LSTM (Iterable): {total_traces} traces en {len(self.batch_files)} archivos")
        print(f"   💾 Memoria: 1 archivo por worker (máx ~{max(self.batch_sizes)} traces)")
        print(f"   🔀 Shuffle: Buffer local de {shuffle_buffer_size} traces")
        print(f"   ⚡ Soporta num_workers > 0 para paralelización")
    
    def __len__(self):
        """Retorna el número total de traces en el dataset"""
        return self.total_traces
    
    def _get_worker_files(self):
        """Divide archivos entre workers"""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Modo single-process
            return self.batch_files
        else:
            # Multi-process: dividir archivos entre workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # Asignar archivos de manera round-robin
            worker_files = [f for i, f in enumerate(self.batch_files) if i % num_workers == worker_id]
            return worker_files
    
    def _shuffle_buffer(self, iterator, buffer_size):
        """Shuffling local mediante buffer"""
        buffer = deque(maxlen=buffer_size)
        
        # Llenar buffer inicial
        for item in iterator:
            buffer.append(item)
            if len(buffer) == buffer_size:
                break
        
        # Yield elementos aleatorios del buffer, rellenando
        for item in iterator:
            idx = random.randint(0, len(buffer) - 1)
            yield buffer[idx]
            buffer[idx] = item
        
        # Vaciar buffer restante
        buffer_list = list(buffer)
        random.shuffle(buffer_list)
        for item in buffer_list:
            yield item
    
    def _generate_samples(self):
        """Genera samples de los archivos asignados al worker"""
        worker_files = self._get_worker_files()
        
        for batch_file in worker_files:
            # Cargar archivo completo
            batch_data = torch.load(batch_file, weights_only=False)
            
            # Yield todas las traces del archivo
            for i in range(len(batch_data['sequences'])):
                yield (
                    batch_data['sequences'][i],
                    batch_data['labels'][i],
                    batch_data['question_ids'][i]
                )
            
            # Liberar memoria inmediatamente
            del batch_data
            gc.collect()
    
    def __iter__(self):
        """Iterador con shuffling local"""
        sample_iterator = self._generate_samples()
        
        if self.shuffle_buffer_size > 0:
            sample_iterator = self._shuffle_buffer(sample_iterator, self.shuffle_buffer_size)
        
        return sample_iterator


class PreprocessedGNNDataset(IterableDataset):
    """
    IterableDataset para GNN-det+LSTM y GVAE+LSTM con soporte para múltiples workers.
    
    ESTRATEGIA:
    - Cada worker procesa un subconjunto de archivos .pt
    - Lee un archivo completo, yield todos los grafos, cierra y pasa al siguiente
    - Shuffling local mediante buffer para mejorar aleatoriedad
    - Permite paralelización real sin cargar todo en memoria
    
    VENTAJAS:
    - ✅ Soporta num_workers > 0 (paralelización real)
    - ✅ Solo 1 archivo en memoria por worker a la vez
    - ✅ Shuffling local aceptable para datasets grandes
    """
    def __init__(self, preprocessed_dir, batch_files_to_load=None, shuffle_buffer_size=1000):
        """
        Args:
            preprocessed_dir: Directorio con archivos preprocessed_*.pt
            batch_files_to_load: Lista de archivos específicos a cargar (None = cargar todos)
            shuffle_buffer_size: Tamaño del buffer para shuffling local (1000 traces ~aceptable)
        """
        super().__init__()
        self.preprocessed_dir = Path(preprocessed_dir)
        self.shuffle_buffer_size = shuffle_buffer_size
        
        # Buscar archivos de batches
        all_batch_files = sorted(list(self.preprocessed_dir.glob('preprocessed_*.pt')))
        if not all_batch_files:
            raise ValueError(f"No se encontraron archivos preprocessed_*.pt en {preprocessed_dir}")
        
        # Determinar qué archivos cargar
        if batch_files_to_load is not None:
            self.batch_files = [Path(f) for f in batch_files_to_load]
        else:
            self.batch_files = all_batch_files
        
        # Contar total de traces (solo metadata, sin cargar datos)
        print(f"📦 Escaneando {len(self.batch_files)} archivos batch (solo metadata)...")
        self.batch_sizes = []
        total_traces = 0
        
        for batch_file in tqdm(self.batch_files, desc="Escaneando"):
            batch_data = torch.load(batch_file, weights_only=False)
            batch_size = len(batch_data['graphs'])
            self.batch_sizes.append(batch_size)
            total_traces += batch_size
            del batch_data
            gc.collect()
        
        self.total_traces = total_traces
        
        print(f"✅ Dataset GNN (Iterable): {total_traces} traces en {len(self.batch_files)} archivos")
        print(f"   💾 Memoria: 1 archivo por worker (máx ~{max(self.batch_sizes)} traces)")
        print(f"   🔀 Shuffle: Buffer local de {shuffle_buffer_size} traces")
        print(f"   ⚡ Soporta num_workers > 0 para paralelización")
    
    def __len__(self):
        """Retorna el número total de traces en el dataset"""
        return self.total_traces
    
    def _get_worker_files(self):
        """Divide archivos entre workers"""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Modo single-process
            return self.batch_files
        else:
            # Multi-process: dividir archivos entre workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # Asignar archivos de manera round-robin
            worker_files = [f for i, f in enumerate(self.batch_files) if i % num_workers == worker_id]
            return worker_files
    
    def _shuffle_buffer(self, iterator, buffer_size):
        """Shuffling local mediante buffer"""
        buffer = deque(maxlen=buffer_size)
        
        # Llenar buffer inicial
        for item in iterator:
            buffer.append(item)
            if len(buffer) == buffer_size:
                break
        
        # Yield elementos aleatorios del buffer, rellenando
        for item in iterator:
            idx = random.randint(0, len(buffer) - 1)
            yield buffer[idx]
            buffer[idx] = item
        
        # Vaciar buffer restante
        buffer_list = list(buffer)
        random.shuffle(buffer_list)
        for item in buffer_list:
            yield item
    
    def _generate_samples(self):
        """Genera samples de los archivos asignados al worker"""
        worker_files = self._get_worker_files()
        
        for batch_file in worker_files:
            # Cargar archivo completo
            batch_data = torch.load(batch_file, weights_only=False)
            
            # Yield todos los grafos del archivo
            for i in range(len(batch_data['graphs'])):
                yield (
                    batch_data['graphs'][i],
                    batch_data['labels'][i],
                    batch_data['question_ids'][i]
                )
            
            # Liberar memoria inmediatamente
            del batch_data
            gc.collect()
    
    def __iter__(self):
        """Iterador con shuffling local"""
        sample_iterator = self._generate_samples()
        
        if self.shuffle_buffer_size > 0:
            sample_iterator = self._shuffle_buffer(sample_iterator, self.shuffle_buffer_size)
        
        return sample_iterator


class SequentialTraceDataset:
    """
    Dataset simplificado que carga archivos .pkl.gz DIRECTAMENTE sin lazy loading complejo.
    Carga solo el número necesario de archivos en memoria.
    
    Aplica threshold sobre scores BLEURT para obtener etiquetas binarias:
    - label = 1 si score < threshold (alucinación)
    - label = 0 si score >= threshold (no alucinación)
    """
    def __init__(self, pkl_files_pattern, scores_file, attn_threshold=0.01, score_threshold=0.5, max_traces_total=None):
        import glob
        import pickle
        import gzip
        
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
        
        # Buscar archivos
        print("Buscando archivos pkl/pkl.gz...")
        file_paths = glob.glob(pkl_files_pattern)
        if not pkl_files_pattern.endswith('.gz') and not pkl_files_pattern.endswith('*'):
            file_paths.extend(glob.glob(pkl_files_pattern.replace('.pkl', '.pkl.gz')))
        
        file_paths = sorted(list(set([f for f in file_paths if not f.endswith('.part')])))
        print(f"Encontrados {len(file_paths)} archivos.")
        
        if not file_paths:
            raise ValueError("No se encontraron archivos .pkl o .pkl.gz")
        
        # Cargar traces directamente en memoria
        self.all_traces = []
        self.attn_threshold = attn_threshold
        traces_loaded = 0
        
        print(f"Cargando archivos .pkl.gz...")
        for file_path in tqdm(file_paths, desc="Cargando"):
            # Si ya tenemos suficientes traces, parar
            if max_traces_total is not None and traces_loaded >= max_traces_total:
                print(f"⚠️  Alcanzado límite de {max_traces_total} traces")
                break
            
            # Cargar archivo
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rb') as f:
                    batch_data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    batch_data = pickle.load(f)
            
            # Determinar cuántos traces tomar de este archivo
            if max_traces_total is not None:
                traces_to_take = min(len(batch_data), max_traces_total - traces_loaded)
                self.all_traces.extend(batch_data[:traces_to_take])
                traces_loaded += traces_to_take
            else:
                self.all_traces.extend(batch_data)
                traces_loaded += len(batch_data)
        
        if not self.all_traces:
            raise ValueError("No se cargaron traces")
        
        self.num_layers = len(self.all_traces[0]['hidden_states'])
        self.num_traces = len(self.all_traces)
        
        print(f"\nDataset creado:")
        print(f"  - {self.num_traces} traces cargados")
        print(f"  - {self.num_layers} capas por trace")
        print(f"  - {len(self.scores_dict)} scores disponibles")
        print(f"  - Score threshold: {score_threshold}")
        
        # Estadísticas de balance de clases
        labels_values = list(self.labels_dict.values())
        num_hallucinations = sum(labels_values)
        num_correct = len(labels_values) - num_hallucinations
        print(f"  - Balance de clases:")
        print(f"    * Alucinaciones (1): {num_hallucinations} ({100*num_hallucinations/len(labels_values):.1f}%)")
        print(f"    * No alucinaciones (0): {num_correct} ({100*num_correct/len(labels_values):.1f}%)")
    
    def __len__(self):
        return self.num_traces
    
    def _trace_to_graph(self, trace, layer_idx):
        """
        Convierte un trace y una capa específica en un grafo de PyG.
        """
        from torch_geometric.data import Data
        
        # Nodos: hidden_states de la capa
        hidden_states = trace['hidden_states'][layer_idx]
        attentions = trace['attentions'][layer_idx]
        
        seq_len, hidden_dim = hidden_states.shape
        
        # Convertir a torch
        if isinstance(hidden_states, np.ndarray):
            x = torch.from_numpy(hidden_states).float()
        else:
            x = hidden_states.float()
        
        # Crear arcos desde matriz de atención
        # Promediar sobre heads
        if isinstance(attentions, np.ndarray):
            attn_avg = torch.from_numpy(attentions).float().mean(dim=0)
        else:
            attn_avg = attentions.float().mean(dim=0)
        
        # Crear arcos donde atención > threshold
        edge_list = []
        edge_weights = []
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j and attn_avg[i, j] > self.attn_threshold:
                    # i está prestando atención a j
                    # Para message passing: información fluye de j -> i
                    # Así i puede agregar información de j (contexto)
                    edge_list.append([j, i])
                    edge_weights.append(attn_avg[i, j].item())
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        # Crear objeto Data de PyG
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            question_id=trace['question_id']
        )
        
        return data
    
    def __getitem__(self, trace_idx):
        """
        Retorna todos los grafos de un trace (una por capa) + etiqueta binaria.
        
        Returns:
            graphs_by_layer: Lista de Data objects
            label: Etiqueta binaria (0 o 1)
            question_id: ID de la pregunta
        """
        trace = self.all_traces[trace_idx]
        question_id = trace['question_id']
        
        # Crear grafos para todas las capas
        graphs_by_layer = []
        for layer_idx in range(self.num_layers):
            graph = self._trace_to_graph(trace, layer_idx)
            graphs_by_layer.append(graph)
        
        # Obtener etiqueta binaria
        label = self.labels_dict.get(question_id, 0)
        
        return graphs_by_layer, torch.tensor(label, dtype=torch.float), question_id


def collate_lstm_batch(batch):
    """
    Collate function para batches de LSTM preprocesados.
    
    Args:
        batch: Lista de (sequence, label, question_id)
    
    Returns:
        sequences: Tensor [batch_size, num_layers, hidden_dim]
        labels: Tensor [batch_size]
        question_ids: Lista de IDs
    """
    sequences = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    question_ids = [item[2] for item in batch]
    
    return sequences, labels, question_ids


def collate_gnn_batch(batch):
    """
    Collate function para batches de GNN preprocesados.
    
    Args:
        batch: Lista de (graphs_by_layer, label, question_id)
        donde graphs_by_layer es una lista de Data objects de PyG
    
    Returns:
        batched_by_layer: Lista de num_layers batches de PyG
        labels: Tensor [batch_size]
        question_ids: Lista de IDs
    """
    from torch_geometric.data import Batch as PyGBatch
    
    if len(batch) == 0:
        raise ValueError("Batch vacío recibido en collate_fn")
    
    num_layers = len(batch[0][0])
    
    # Reorganizar: en lugar de [batch][layer], queremos [layer][batch]
    batched_by_layer = []
    for layer_idx in range(num_layers):
        layer_graphs = [item[0][layer_idx] for item in batch]
        batched_layer = PyGBatch.from_data_list(layer_graphs)
        batched_by_layer.append(batched_layer)
    
    labels = torch.stack([item[1] for item in batch])
    question_ids = [item[2] for item in batch]
    
    return batched_by_layer, labels, question_ids


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
# Funciones auxiliares para evaluación
# ============================================================================

def find_optimal_threshold(labels, probs):
    """
    Encuentra el threshold óptimo maximizando Youden's J statistic (sensitivity + specificity - 1).
    Este threshold es óptimo para AUROC ya que maximiza la distancia a la línea diagonal en la curva ROC.
    
    También conocido como el punto más cercano a (0,1) en la curva ROC.
    """
    from sklearn.metrics import roc_curve
    
    # Calcular curva ROC
    fpr, tpr, thresholds = roc_curve(labels, probs)
    
    # Youden's J statistic = Sensitivity + Specificity - 1
    # Sensitivity = TPR (True Positive Rate)
    # Specificity = 1 - FPR (False Positive Rate)
    # J = TPR + (1 - FPR) - 1 = TPR - FPR
    youdens_j = tpr - fpr
    
    # Encontrar el threshold que maximiza J
    optimal_idx = np.argmax(youdens_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calcular F1 en el threshold óptimo para reportar
    optimal_preds = (probs >= optimal_threshold).astype(int)
    optimal_f1 = f1_score(labels, optimal_preds, zero_division=0)
    
    return optimal_threshold, optimal_f1


def evaluate_model(model, data_loader, device, threshold=0.5, is_gvae=False):
    """
    Evalúa un modelo en un conjunto de datos y retorna métricas.
    Compatible con LSTM, GNN-det y GVAE.
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            batched_by_layer, labels, _ = batch_data
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            
            # Detectar tipo de datos y procesar según modelo
            if isinstance(batched_by_layer, torch.Tensor):
                # Datos preprocesados LSTM
                layer_sequence = batched_by_layer.to(device, dtype=torch.float32, non_blocking=True)
                logits = model(layer_sequence)
            else:
                # Datos raw (grafos) - convertir a float32 para evitar dtype mismatch
                batched_by_layer_gpu = []
                for layer_data in batched_by_layer:
                    layer_data = layer_data.to(device, non_blocking=True)
                    # Convertir atributos a float32 si son half
                    if layer_data.x.dtype == torch.float16:
                        layer_data.x = layer_data.x.to(torch.float32)
                    if hasattr(layer_data, 'edge_attr') and layer_data.edge_attr is not None and layer_data.edge_attr.dtype == torch.float16:
                        layer_data.edge_attr = layer_data.edge_attr.to(torch.float32)
                    batched_by_layer_gpu.append(layer_data)
                
                if is_gvae:
                    # GVAE devuelve múltiples salidas
                    logits, _, _, _, _ = model(batched_by_layer_gpu, len(batched_by_layer_gpu))
                else:
                    # GNN-det o LSTM desde grafos
                    logits = model(batched_by_layer_gpu, len(batched_by_layer_gpu))
            
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Calcular métricas
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > threshold).astype(float)
    
    metrics = {
        'auroc': roc_auc_score(all_labels, all_probs),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'threshold': threshold,
        'probs': all_probs,
        'labels': all_labels,
        'preds': all_preds
    }
    
    return metrics


# ============================================================================
# Funciones de entrenamiento
# ============================================================================

def train_lstm_baseline(model, train_loader, val_loader, test_loader, device, epochs=50, lr=0.001):
    """
    Entrena el modelo LSTM Baseline con clasificación binaria.
    OPTIMIZADO PARA MEMORIA: Libera memoria GPU/RAM regularmente.
    
    Compatible con:
    - Datos preprocesados (collate_lstm_batch): recibe tensores directamente
    - Datos raw (collate_sequential_batch): recibe grafos PyG
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()  # BCE con logits (más estable numéricamente)
    
    history = {'train_loss': [], 'val_loss': [], 'val_auroc': [], 'val_acc': [], 'val_f1': []}
    best_val_auroc = 0.0
    best_threshold = 0.5  # Inicializar threshold
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batched_by_layer, labels, _ = batch_data
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            
            # Detectar si son datos preprocesados (tensores) o raw (grafos PyG)
            if isinstance(batched_by_layer, torch.Tensor):
                # Datos preprocesados: ya tenemos las secuencias listas
                # Convertir a float32 si es necesario (preprocesamiento usa float16)
                layer_sequence = batched_by_layer.to(device, dtype=torch.float32, non_blocking=True)
            else:
                # Datos raw: extraer secuencia de grafos
                # OPTIMIZACIÓN: Transferir todas las capas a GPU de una vez
                batched_by_layer_gpu = []
                for layer_data in batched_by_layer:
                    layer_data_gpu = layer_data.to(device, non_blocking=True)
                    batched_by_layer_gpu.append(layer_data_gpu)
                
                # Extraer secuencia: SOLO ÚLTIMO TOKEN (sin información estructural)
                # Esto es clave para LSTM-solo: comparar el último token a través de capas
                layer_sequence = []
                for layer_data in batched_by_layer_gpu:
                    # Extraer hidden states del último token de cada grafo en el batch
                    batch_size = layer_data.batch.max().item() + 1
                    last_token_features = []
                    
                    for batch_idx in range(batch_size):
                        # Obtener nodos de este grafo en el batch
                        node_mask = (layer_data.batch == batch_idx)
                        graph_nodes = layer_data.x[node_mask]
                        
                        # Tomar el ÚLTIMO nodo (último token)
                        last_token = graph_nodes[-1]  # [hidden_dim]
                        last_token_features.append(last_token)
                    
                    # Stack para crear batch
                    layer_repr = torch.stack(last_token_features, dim=0)  # [batch, hidden_dim]
                    layer_sequence.append(layer_repr)
                
                layer_sequence = torch.stack(layer_sequence, dim=1)  # [batch, layers, dim]
            
            optimizer.zero_grad()
            logits = model(layer_sequence)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Liberar memoria
            del layer_sequence, logits, loss, labels
            if not isinstance(batched_by_layer, torch.Tensor):
                del batched_by_layer_gpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Liberar memoria después del training
        gc.collect()
        
        # Validation
        model.eval()
        val_loss = 0
        all_probs = []
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                batched_by_layer, labels, _ = batch_data
                labels = labels.to(device, non_blocking=True).unsqueeze(1)
                
                # Detectar si son datos preprocesados (tensores) o raw (grafos PyG)
                if isinstance(batched_by_layer, torch.Tensor):
                    # Datos preprocesados: ya tenemos las secuencias listas
                    # Convertir a float32 si es necesario (preprocesamiento usa float16)
                    layer_sequence = batched_by_layer.to(device, dtype=torch.float32, non_blocking=True)
                else:
                    # Datos raw: extraer secuencia de grafos
                    # OPTIMIZACIÓN: Transferir todas las capas a GPU de una vez
                    batched_by_layer_gpu = []
                    for layer_data in batched_by_layer:
                        layer_data_gpu = layer_data.to(device, non_blocking=True)
                        batched_by_layer_gpu.append(layer_data_gpu)
                    
                    # Extraer secuencia: SOLO ÚLTIMO TOKEN (sin información estructural)
                    layer_sequence = []
                    for layer_data in batched_by_layer_gpu:
                        # Extraer hidden states del último token de cada grafo en el batch
                        batch_size = layer_data.batch.max().item() + 1
                        last_token_features = []
                        
                        for batch_idx in range(batch_size):
                            # Obtener nodos de este grafo en el batch
                            node_mask = (layer_data.batch == batch_idx)
                            graph_nodes = layer_data.x[node_mask]
                            
                            # Tomar el ÚLTIMO nodo (último token)
                            last_token = graph_nodes[-1]  # [hidden_dim]
                            last_token_features.append(last_token)
                        
                        # Stack para crear batch
                        layer_repr = torch.stack(last_token_features, dim=0)  # [batch, hidden_dim]
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
                
                # Liberar memoria
                del layer_sequence, logits, loss, probs, preds, labels
                if not isinstance(batched_by_layer, torch.Tensor):
                    del batched_by_layer_gpu
        
        # Liberar memoria después de validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Encontrar threshold óptimo en validación
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        optimal_threshold, _ = find_optimal_threshold(all_labels, all_probs)
        
        # Recalcular predicciones con threshold óptimo
        all_preds_optimal = (all_probs > optimal_threshold).astype(float)
        
        # Calcular métricas con threshold óptimo
        val_auroc = roc_auc_score(all_labels, all_probs)
        val_acc = accuracy_score(all_labels, all_preds_optimal)
        val_f1 = f1_score(all_labels, all_preds_optimal, zero_division=0)
        val_precision = precision_score(all_labels, all_preds_optimal, zero_division=0)
        val_recall = recall_score(all_labels, all_preds_optimal, zero_division=0)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"AUROC={val_auroc:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f} (thr={optimal_threshold:.3f})")
        
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_threshold = optimal_threshold
            torch.save(model.state_dict(), 'best_lstm_baseline.pt')
    
    # Evaluación final en conjunto de test
    print("\n" + "="*80)
    print("EVALUACIÓN EN CONJUNTO DE TEST")
    print("="*80)
    model.load_state_dict(torch.load('best_lstm_baseline.pt', weights_only=False))
    test_metrics = evaluate_model(model, test_loader, device, threshold=best_threshold, is_gvae=False)
    
    print(f"\nMétricas en TEST (threshold={best_threshold:.3f}):")
    print(f"  AUROC:     {test_metrics['auroc']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    
    # Agregar métricas de test al historial
    history['test_auroc'] = test_metrics['auroc']
    history['test_acc'] = test_metrics['accuracy']
    history['test_f1'] = test_metrics['f1']
    history['test_precision'] = test_metrics['precision']
    history['test_recall'] = test_metrics['recall']
    history['best_threshold'] = best_threshold
    
    return history


def train_gnn_det_lstm(model, train_loader, val_loader, test_loader, device, epochs=50, lr=0.001):
    """
    Entrena el modelo GNN-det+LSTM con clasificación binaria.
    OPTIMIZADO PARA MEMORIA: Libera memoria GPU/RAM regularmente.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_auroc': [], 'val_acc': [], 'val_f1': []}
    best_val_auroc = 0.0
    best_threshold = 0.5  # Inicializar threshold
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batched_by_layer, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            labels = labels.to(device).unsqueeze(1)
            
            # Mover datos a device y convertir a float32 si es necesario
            batched_by_layer_gpu = []
            for data in batched_by_layer:
                data = data.to(device)
                # Convertir atributos a float32 si son half
                if data.x.dtype == torch.float16:
                    data.x = data.x.to(torch.float32)
                if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.dtype == torch.float16:
                    data.edge_attr = data.edge_attr.to(torch.float32)
                batched_by_layer_gpu.append(data)
            
            optimizer.zero_grad()
            logits = model(batched_by_layer_gpu, len(batched_by_layer_gpu))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Liberar memoria
            del logits, loss, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Liberar memoria después del training
        gc.collect()
        
        model.eval()
        val_loss = 0
        all_probs = []
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batched_by_layer, labels, _ in val_loader:
                labels = labels.to(device).unsqueeze(1)
                # Convertir a float32 para evitar dtype mismatch
                batched_by_layer_gpu = []
                for data in batched_by_layer:
                    data = data.to(device)
                    # Convertir atributos a float32 si son half
                    if data.x.dtype == torch.float16:
                        data.x = data.x.to(torch.float32)
                    if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.dtype == torch.float16:
                        data.edge_attr = data.edge_attr.to(torch.float32)
                    batched_by_layer_gpu.append(data)
                
                logits = model(batched_by_layer_gpu, len(batched_by_layer_gpu))
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Calcular probabilidades y predicciones
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                
                # Liberar memoria
                del logits, loss, probs, preds, labels
        
        # Liberar memoria después de validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Encontrar threshold óptimo en validación (Youden's J para AUROC)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        optimal_threshold, _ = find_optimal_threshold(all_labels, all_probs)
        
        # Recalcular predicciones con threshold óptimo
        all_preds_optimal = (all_probs > optimal_threshold).astype(float)
        
        # Calcular métricas con threshold óptimo
        val_auroc = roc_auc_score(all_labels, all_probs)
        val_acc = accuracy_score(all_labels, all_preds_optimal)
        val_f1 = f1_score(all_labels, all_preds_optimal, zero_division=0)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"AUROC={val_auroc:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f} (thr={optimal_threshold:.3f})")
        
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_threshold = optimal_threshold
            torch.save(model.state_dict(), 'best_gnn_det_lstm.pt')
    
    # Evaluación final en conjunto de test
    print("\n" + "="*80)
    print("EVALUACIÓN EN CONJUNTO DE TEST")
    print("="*80)
    model.load_state_dict(torch.load('best_gnn_det_lstm.pt', weights_only=False))
    test_metrics = evaluate_model(model, test_loader, device, threshold=best_threshold, is_gvae=False)
    
    print(f"\nMétricas en TEST (threshold={best_threshold:.3f}):")
    print(f"  AUROC:     {test_metrics['auroc']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    
    # Agregar métricas de test al historial
    history['test_auroc'] = test_metrics['auroc']
    history['test_acc'] = test_metrics['accuracy']
    history['test_f1'] = test_metrics['f1']
    history['test_precision'] = test_metrics['precision']
    history['test_recall'] = test_metrics['recall']
    history['best_threshold'] = best_threshold
    
    return history


def train_gvae_lstm(model, train_loader, val_loader, test_loader, device, epochs=50, lr=0.001, kl_weight=0.001):
    """
    Entrena el modelo GVAE+LSTM con clasificación binaria.
    OPTIMIZADO PARA MEMORIA: Libera memoria GPU/RAM regularmente.
    
    NOTA: Guarda loss separadas para comparación justa:
    - train_task_loss: Solo loss de clasificación (comparable con otros modelos)
    - train_vae_loss: Loss del autoencoder
    - train_loss: Loss total (task + VAE)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    history = {'train_loss': [], 'train_task_loss': [], 'train_vae_loss': [],
               'val_loss': [], 'val_auroc': [], 'val_acc': [], 'val_f1': []}
    best_val_auroc = 0.0
    best_threshold = 0.5  # Inicializar threshold
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_task_loss = 0
        train_vae_loss = 0
        
        for batched_by_layer, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            labels = labels.to(device).unsqueeze(1)
            # Convertir a float32 para evitar dtype mismatch
            batched_by_layer_gpu = []
            for data in batched_by_layer:
                data = data.to(device)
                # Convertir atributos a float32 si son half
                if data.x.dtype == torch.float16:
                    data.x = data.x.to(torch.float32)
                if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.dtype == torch.float16:
                    data.edge_attr = data.edge_attr.to(torch.float32)
                batched_by_layer_gpu.append(data)
            
            optimizer.zero_grad()
            logits, mu_list, logvar_list, orig_list, recon_list = model(
                batched_by_layer_gpu, len(batched_by_layer_gpu)
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
            
            # Liberar memoria
            del logits, mu_list, logvar_list, orig_list, recon_list
            del task_loss, vae_loss_total, loss, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Liberar memoria después del training
        gc.collect()
        
        model.eval()
        val_loss = 0
        all_probs = []
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batched_by_layer, labels, _ in val_loader:
                labels = labels.to(device).unsqueeze(1)
                # Convertir a float32 para evitar dtype mismatch
                batched_by_layer_gpu = []
                for data in batched_by_layer:
                    data = data.to(device)
                    # Convertir atributos a float32 si son half
                    if data.x.dtype == torch.float16:
                        data.x = data.x.to(torch.float32)
                    if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.dtype == torch.float16:
                        data.edge_attr = data.edge_attr.to(torch.float32)
                    batched_by_layer_gpu.append(data)
                
                logits, _, _, _, _ = model(batched_by_layer_gpu, len(batched_by_layer_gpu))
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Calcular probabilidades y predicciones
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                
                # Liberar memoria
                del logits, loss, probs, preds, labels
        
        # Liberar memoria después de validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        train_loss /= len(train_loader)
        train_task_loss /= len(train_loader)
        train_vae_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Encontrar threshold óptimo en validación (Youden's J para AUROC)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        optimal_threshold, _ = find_optimal_threshold(all_labels, all_probs)
        
        # Recalcular predicciones con threshold óptimo
        all_preds_optimal = (all_probs > optimal_threshold).astype(float)
        
        # Calcular métricas con threshold óptimo
        val_auroc = roc_auc_score(all_labels, all_probs)
        val_acc = accuracy_score(all_labels, all_preds_optimal)
        val_f1 = f1_score(all_labels, all_preds_optimal, zero_division=0)
        
        history['train_loss'].append(train_loss)
        history['train_task_loss'].append(train_task_loss)
        history['train_vae_loss'].append(train_vae_loss)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} (Task={train_task_loss:.4f}, VAE={train_vae_loss:.4f}), "
              f"Val Loss={val_loss:.4f}, AUROC={val_auroc:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f} (thr={optimal_threshold:.3f})")
        
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_threshold = optimal_threshold
            torch.save(model.state_dict(), 'best_gvae_lstm.pt')
    
    # Evaluación final en conjunto de test
    print("\n" + "="*80)
    print("EVALUACIÓN EN CONJUNTO DE TEST")
    print("="*80)
    model.load_state_dict(torch.load('best_gvae_lstm.pt', weights_only=False))
    test_metrics = evaluate_model(model, test_loader, device, threshold=best_threshold, is_gvae=True)
    
    print(f"\nMétricas en TEST (threshold={best_threshold:.3f}):")
    print(f"  AUROC:     {test_metrics['auroc']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    
    # Agregar métricas de test al historial
    history['test_auroc'] = test_metrics['auroc']
    history['test_acc'] = test_metrics['accuracy']
    history['test_f1'] = test_metrics['f1']
    history['test_precision'] = test_metrics['precision']
    history['test_recall'] = test_metrics['recall']
    history['best_threshold'] = best_threshold
    
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
    
    # Determinar si usar datos preprocesados o raw
    use_preprocessed = hasattr(args, 'preprocessed_dir') and args.preprocessed_dir is not None
    
    from torch.utils.data import DataLoader
    import os
    
    if use_preprocessed:
        print(f"📦 Usando datos preprocesados desde: {args.preprocessed_dir}")
        
        # Obtener lista de archivos batch
        lstm_dir = Path(args.preprocessed_dir) / 'lstm_solo'
        gnn_dir = Path(args.preprocessed_dir) / 'gnn'
        
        all_lstm_files = sorted(list(lstm_dir.glob('preprocessed_*.pt')))
        all_gnn_files = sorted(list(gnn_dir.glob('preprocessed_*.pt')))
        
        if not all_lstm_files or not all_gnn_files:
            raise ValueError("No se encontraron archivos preprocesados")
        
        # Split archivos a nivel de archivo (70% train, 15% val, 15% test)
        # Esto asegura que cada subset tenga archivos completos
        random.seed(42)
        lstm_files_shuffled = all_lstm_files.copy()
        gnn_files_shuffled = all_gnn_files.copy()
        random.shuffle(lstm_files_shuffled)
        random.shuffle(gnn_files_shuffled)
        
        # Usar el número de archivos de cada carpeta para el split
        n_lstm_files = len(lstm_files_shuffled)
        n_gnn_files = len(gnn_files_shuffled)
        
        # Split para LSTM
        train_split_lstm = int(0.7 * n_lstm_files)
        val_split_lstm = int(0.85 * n_lstm_files)
        
        train_lstm_files = lstm_files_shuffled[:train_split_lstm]
        val_lstm_files = lstm_files_shuffled[train_split_lstm:val_split_lstm]
        test_lstm_files = lstm_files_shuffled[val_split_lstm:]
        
        # Split para GNN
        train_split_gnn = int(0.7 * n_gnn_files)
        val_split_gnn = int(0.85 * n_gnn_files)
        
        train_gnn_files = gnn_files_shuffled[:train_split_gnn]
        val_gnn_files = gnn_files_shuffled[train_split_gnn:val_split_gnn]
        test_gnn_files = gnn_files_shuffled[val_split_gnn:]
        
        print(f"💾 Estrategia: IterableDataset con múltiples workers")
        print(f"   - Archivos LSTM: {len(train_lstm_files)} train, {len(val_lstm_files)} val, {len(test_lstm_files)} test")
        print(f"   - Archivos GNN: {len(train_gnn_files)} train, {len(val_gnn_files)} val, {len(test_gnn_files)} test")
        print(f"   ⚡ Soporta num_workers > 0 para paralelización")
        print(f"   🔀 Shuffling local con buffer de 500 traces")
        
        # Crear datasets separados para train/val/test
        train_dataset_lstm = PreprocessedLSTMDataset(lstm_dir, batch_files_to_load=train_lstm_files, shuffle_buffer_size=250)
        val_dataset_lstm = PreprocessedLSTMDataset(lstm_dir, batch_files_to_load=val_lstm_files, shuffle_buffer_size=0)
        test_dataset_lstm = PreprocessedLSTMDataset(lstm_dir, batch_files_to_load=test_lstm_files, shuffle_buffer_size=0)
        
        train_dataset_gnn = PreprocessedGNNDataset(gnn_dir, batch_files_to_load=train_gnn_files, shuffle_buffer_size=250)
        val_dataset_gnn = PreprocessedGNNDataset(gnn_dir, batch_files_to_load=val_gnn_files, shuffle_buffer_size=0)
        test_dataset_gnn = PreprocessedGNNDataset(gnn_dir, batch_files_to_load=test_gnn_files, shuffle_buffer_size=0)
        
        # Determinar num_workers óptimo
        # Regla: num_workers = min(num_archivos_train, num_cpus, 4)
        import multiprocessing
        num_cpus = multiprocessing.cpu_count()
        num_workers = min(len(train_lstm_files), num_cpus, 2)
        
        print(f"\nConfigurando DataLoaders:")
        print(f"  - num_workers: {num_workers} (paralelización real)")
        print(f"  - pin_memory: {device.type == 'cuda'}")
        print(f"  - Memoria: ~{num_workers} archivos batch en memoria simultáneos")
        print(f"  ⚡ Cada worker procesa archivos diferentes en paralelo")
        
        # DataLoaders para LSTM (sin shuffle=True porque IterableDataset ya hace shuffle interno)
        train_loader_lstm = DataLoader(
            train_dataset_lstm,
            batch_size=args.batch_size,
            collate_fn=collate_lstm_batch,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda'
        )
        val_loader_lstm = DataLoader(
            val_dataset_lstm,
            batch_size=args.batch_size,
            collate_fn=collate_lstm_batch,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda'
        )
        test_loader_lstm = DataLoader(
            test_dataset_lstm,
            batch_size=args.batch_size,
            collate_fn=collate_lstm_batch,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda'
        )
        
        # DataLoaders para GNN (usados por GNN-det+LSTM y GVAE+LSTM)
        train_loader_gnn = DataLoader(
            train_dataset_gnn,
            batch_size=args.batch_size,
            collate_fn=collate_gnn_batch,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda'
        )
        val_loader_gnn = DataLoader(
            val_dataset_gnn,
            batch_size=args.batch_size,
            collate_fn=collate_gnn_batch,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda'
        )
        test_loader_gnn = DataLoader(
            test_dataset_gnn,
            batch_size=args.batch_size,
            collate_fn=collate_gnn_batch,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda'
        )
        
        # Obtener dimensiones (necesitamos iterar para obtener el primer sample)
        print("\nObteniendo dimensiones del dataset...")
        for sample_seq, _, _ in train_dataset_lstm:
            hidden_dim = sample_seq.shape[-1]
            print(f"Dimensión de hidden states: {hidden_dim}")
            break
        
    else:
        print("📂 Usando datos raw (archivos .pkl/.pkl.gz)")
        
        # Si se especifica max_traces, limitamos el número total de traces
        max_traces_total = args.max_traces if args.max_traces is not None else None
        
        if max_traces_total is not None:
            print(f"⚠️  Limitando a {max_traces_total} traces totales")
        
        full_dataset = SequentialTraceDataset(
            args.data_pattern,
            args.scores_file,
            attn_threshold=args.attn_threshold,
            score_threshold=args.score_threshold,
            max_traces_total=max_traces_total
        )
        
        # Split train/val/test
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"\nSplit del dataset:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Val: {len(val_dataset)}")
        print(f"  Test: {len(test_dataset)}")
        
        # Configurar num_workers
        num_workers = min(4, os.cpu_count() or 1)
        
        print(f"\nConfigurando DataLoaders:")
        print(f"  - num_workers: {num_workers} (prefetching paralelo)")
        print(f"  - pin_memory: True (transferencias GPU más rápidas)")
        
        train_loader_lstm = train_loader_gnn = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True, 
            collate_fn=collate_sequential_batch,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda',
            persistent_workers=num_workers > 0
        )
        val_loader_lstm = val_loader_gnn = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            shuffle=False, 
            collate_fn=collate_sequential_batch,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda'
        )
        test_loader_lstm = test_loader_gnn = DataLoader(
            test_dataset, 
            batch_size=args.batch_size,
            shuffle=False, 
            collate_fn=collate_sequential_batch,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda'
        )
        
        # Obtener dimensiones
        sample_graph = full_dataset[0][0][0]
        hidden_dim = sample_graph.x.shape[1]
        print(f"\nDimensión de hidden states: {hidden_dim}")
    
    results = {}
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Función auxiliar para convertir tipos numpy/torch a tipos nativos Python
    def convert_to_serializable(obj):
        """Convierte numpy/torch types a tipos nativos de Python para JSON"""
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # torch tensors
            return obj.item()
        else:
            return obj
    
    # Función auxiliar para guardar resultados parciales
    def save_partial_results(model_name, metrics, config):
        """Guarda resultados de un modelo inmediatamente después de entrenar"""
        partial_file = output_dir / f"partial_{model_name.lower().replace('+', '_').replace('-', '_')}_{timestamp}.json"
        
        partial_results = {
            'model': model_name,
            'metrics': {
                'best_val_auroc': metrics['best_val_auroc'],
                'best_val_acc': metrics['best_val_acc'],
                'best_val_f1': metrics['best_val_f1'],
                'test_auroc': metrics.get('test_auroc', 0.0),
                'test_acc': metrics.get('test_acc', 0.0),
                'test_f1': metrics.get('test_f1', 0.0),
                'best_threshold': metrics.get('best_threshold', 0.5),
                'history': metrics['history']
            },
            'config': config,
            'timestamp': timestamp
        }
        
        # Convertir a tipos serializables
        partial_results = convert_to_serializable(partial_results)
        
        with open(partial_file, 'w') as f:
            json.dump(partial_results, f, indent=2)
        
        print(f"\n✅ Resultados de {model_name} guardados en: {partial_file}")
        return partial_file
    
    # Configuración compartida para todos los modelos
    shared_config = {
        'score_threshold': args.score_threshold,
        'attn_threshold': args.attn_threshold if hasattr(args, 'attn_threshold') else None,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'gnn_hidden': args.gnn_hidden,
        'latent_dim': args.latent_dim,
        'lstm_hidden': args.lstm_hidden,
        'num_lstm_layers': args.num_lstm_layers,
        'dropout': args.dropout,
        'kl_weight': args.kl_weight,
    }
    
    # ========================================================================
    # EXPERIMENTO 1: LSTM Baseline
    # ========================================================================
    if not args.skip_lstm:
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
        
        try:
            history_lstm = train_lstm_baseline(
                model_lstm, train_loader_lstm, val_loader_lstm, test_loader_lstm, device,
                epochs=args.epochs, lr=args.lr
            )
            
            results['LSTM-solo'] = {
                'best_val_auroc': max(history_lstm['val_auroc']),
                'best_val_acc': max(history_lstm['val_acc']),
                'best_val_f1': max(history_lstm['val_f1']),
                'test_auroc': history_lstm.get('test_auroc', 0.0),
                'test_acc': history_lstm.get('test_acc', 0.0),
                'test_f1': history_lstm.get('test_f1', 0.0),
                'best_threshold': history_lstm.get('best_threshold', 0.5),
                'history': history_lstm
            }
            
            # Guardar inmediatamente después de entrenar
            save_partial_results('LSTM-solo', results['LSTM-solo'], shared_config)
            
            # Liberar memoria del modelo
            del model_lstm, history_lstm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"\n❌ Error durante entrenamiento de LSTM-solo: {e}")
            import traceback
            traceback.print_exc()
            # Continuar con el siguiente modelo
    
    # ========================================================================
    # EXPERIMENTO 2: GNN-det+LSTM
    # ========================================================================
    if not args.skip_gnn_det:
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
        
        try:
            history_gnn = train_gnn_det_lstm(
                model_gnn, train_loader_gnn, val_loader_gnn, test_loader_gnn, device,
                epochs=args.epochs, lr=args.lr
            )
            
            results['GNN-det+LSTM'] = {
                'best_val_auroc': max(history_gnn['val_auroc']),
                'best_val_acc': max(history_gnn['val_acc']),
                'best_val_f1': max(history_gnn['val_f1']),
                'test_auroc': history_gnn.get('test_auroc', 0.0),
                'test_acc': history_gnn.get('test_acc', 0.0),
                'test_f1': history_gnn.get('test_f1', 0.0),
                'best_threshold': history_gnn.get('best_threshold', 0.5),
                'history': history_gnn
            }
            
            # Guardar inmediatamente después de entrenar
            save_partial_results('GNN-det+LSTM', results['GNN-det+LSTM'], shared_config)
            
            # Liberar memoria del modelo
            del model_gnn, history_gnn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"\n❌ Error durante entrenamiento de GNN-det+LSTM: {e}")
            import traceback
            traceback.print_exc()
            # Continuar con el siguiente modelo
    
    # ========================================================================
    # EXPERIMENTO 3: GVAE+LSTM
    # ========================================================================
    if not args.skip_gvae:
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
        
        try:
            history_gvae = train_gvae_lstm(
                model_gvae, train_loader_gnn, val_loader_gnn, test_loader_gnn, device,
                epochs=args.epochs, lr=args.lr, kl_weight=args.kl_weight
            )
            
            results['GVAE+LSTM'] = {
                'best_val_auroc': max(history_gvae['val_auroc']),
                'best_val_acc': max(history_gvae['val_acc']),
                'best_val_f1': max(history_gvae['val_f1']),
                'test_auroc': history_gvae.get('test_auroc', 0.0),
                'test_acc': history_gvae.get('test_acc', 0.0),
                'test_f1': history_gvae.get('test_f1', 0.0),
                'best_threshold': history_gvae.get('best_threshold', 0.5),
                'history': history_gvae
            }
            
            # Guardar inmediatamente después de entrenar
            save_partial_results('GVAE+LSTM', results['GVAE+LSTM'], shared_config)
            
            # Liberar memoria del modelo
            del model_gvae, history_gvae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"\n❌ Error durante entrenamiento de GVAE+LSTM: {e}")
            import traceback
            traceback.print_exc()
            # Continuar (o terminar si es el último modelo)
    
    # ========================================================================
    # RESULTADOS Y CONCLUSIONES
    # ========================================================================
    print("\n" + "="*80)
    print("RESULTADOS FINALES - TABLA DE ABLACIÓN")
    print("="*80)
    
    if len(results) == 0:
        print("\n⚠️  No se entrenó ningún modelo exitosamente.")
        print("Revisa los archivos parciales en el directorio de salida para más detalles.")
        return
    
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
    else:
        print(f"\n⚠️  Solo {len(results)} modelo(s) entrenado(s) exitosamente.")
        print("No se puede realizar comparación completa de ablación.")
        print("\nResultados parciales disponibles en:")
        for model_name in results.keys():
            partial_name = model_name.lower().replace('+', '_').replace('-', '_')
            print(f"  - partial_{partial_name}_{timestamp}.json")
    
    # Guardar resultados finales consolidados (solo si hay resultados)
    if len(results) > 0:
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
        results_json['config'] = shared_config
        
        # Convertir a tipos serializables
        results_json = convert_to_serializable(results_json)
        
        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n✅ Resultados consolidados guardados en: {results_file}")
    
    print("\n" + "="*80)
    print("ARCHIVOS GUARDADOS:")
    print("="*80)
    for model_name in results.keys():
        partial_name = model_name.lower().replace('+', '_').replace('-', '_')
        print(f"  ✓ partial_{partial_name}_{timestamp}.json")
    if len(results) > 0:
        print(f"  ✓ ablation_results_{timestamp}.json (consolidado)")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Experimentos de ablación para detección de alucinaciones (Metodología HaloScope)"
    )
    
    # Datos
    parser.add_argument('--preprocessed-dir', type=str, default=None,
                       help='Directorio con datos preprocesados (si se especifica, se ignoran --data-pattern y --attn-threshold)')
    parser.add_argument('--max-cache-batches', type=int, default=2,
                       help='Número máximo de batches a mantener en cache de memoria (default: 2)')
    parser.add_argument('--data-pattern', type=str, default=None,
                       help='Patrón glob para archivos .pkl o .pkl.gz (ej: "traces_data/*.pkl*") [Solo si no se usa --preprocessed-dir]')
    parser.add_argument('--scores-file', type=str, default=None,
                       help='Archivo CSV con scores BLEURT [Solo si no se usa --preprocessed-dir]')
    parser.add_argument('--attn-threshold', type=float, default=0.0,
                       help='Umbral de atención para crear arcos [Solo para datos raw]')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                       help='Umbral de score BLEURT para etiquetar alucinaciones (score < threshold = alucinación)')
    parser.add_argument('--max-traces', type=int, default=None,
                       help='Número máximo de traces a cargar (None = todos) [Solo para datos raw]')
    
    # Entrenamiento
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Learning rate')
    
    # Arquitectura
    parser.add_argument('--gnn-hidden', type=int, default=128,
                       help='Dimensión oculta de GNN')
    parser.add_argument('--latent-dim', type=int, default=128,
                       help='Dimensión latente para GVAE')
    parser.add_argument('--lstm-hidden', type=int, default=64,
                       help='Dimensión oculta de LSTM')
    parser.add_argument('--num-lstm-layers', type=int, default=2,
                       help='Número de capas LSTM')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--kl-weight', type=float, default=0.001,
                       help='Peso para pérdida KL en GVAE')
    
    # Control de experimentos
    parser.add_argument('--skip-lstm', action='store_true', default=False,
                       help='Saltar experimento LSTM-solo')
    parser.add_argument('--skip-gnn-det', action='store_true', default=False,
                       help='Saltar experimento GNN-det+LSTM')
    parser.add_argument('--skip-gvae', action='store_true', default=False,
                       help='Saltar experimento GVAE+LSTM')
    parser.add_argument('--output-dir', type=str, default='./ablation_results',
                       help='Directorio para guardar resultados')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Forzar ejecución en CPU (útil si hay problemas con CUDA)')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if args.preprocessed_dir is None:
        if args.data_pattern is None or args.scores_file is None:
            parser.error("Debe especificar --preprocessed-dir o ambos (--data-pattern y --scores-file)")
    
    run_ablation_experiments(args)
