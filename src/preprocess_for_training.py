"""
Script de pre-procesamiento para acelerar training.

Este script:
1. Carga traces desde archivos .pkl/.pkl.gz (por batch)
2. Genera 3 versiones optimizadas:
   - LSTM-solo: hidden state del último token por capa
   - GNN-det+LSTM: grafos con atenciones promediadas por capa (respeta causalidad)
   - GVAE: misma estructura que GNN-det+LSTM
3. Guarda archivos separados por batch en formato .pt
4. Usa float16 (consistente con cuantización 4-bit del modelo)

Ventajas:
- Reduce tiempo de carga de ~30 seg a <1 seg por batch
- Archivos resultantes son ~10-20× más pequeños (float16 vs float32)
- Una vez procesado, el entrenamiento es 50-100× más rápido
- Respeta estructura causal: tokens no atienden al futuro

Uso:
    python preprocess_for_training.py \
        --data-pattern "traces_data/*.pkl*" \
        --scores-file ground_truth_scores.csv \
        --output-dir preprocessed_data \
        --attn-threshold 0.0 \
        --score-threshold 0.5
"""

import torch
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc
import re
import glob
import pickle
import gzip
from torch_geometric.data import Data
from dataloader import TraceGraphDataset


def extract_batch_number(filename):
    """Extrae el número de batch del nombre del archivo."""
    match = re.search(r'batch_(\d+)', str(filename))
    if match:
        return int(match.group(1))
    return None


def load_trace_file(file_path):
    """Carga un archivo .pkl o .pkl.gz"""
    if str(file_path).endswith('.gz'):
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(file_path, 'rb') as f:
            return pickle.load(f)


def build_graph_from_trace(trace_data, layer_idx, attn_threshold=0.0):
    """
    Construye un grafo PyG desde los datos del trace para una capa específica.
    Retorna el grafo con atenciones promediadas entre todas las cabezas.
    
    IMPORTANTE: Respeta estructura causal (tokens pasados no atienden al futuro).
    avg_attention[i, j] = cuánto el token i atiende al token j
    Solo creamos arcos donde j <= i (token i solo puede atender a tokens pasados/actuales)
    """
    attention = trace_data['attentions'][layer_idx]  # [num_heads, seq_len, seq_len]
    hidden_states = trace_data['hidden_states'][layer_idx]  # [seq_len, hidden_dim]
    tokens = trace_data['tokens']
    
    # Convertir a tensores si son numpy arrays (usar float16 para consistencia con cuantización 4-bit)
    import numpy as np
    if isinstance(attention, np.ndarray):
        attention = torch.from_numpy(attention).half()  # float16
    if isinstance(hidden_states, np.ndarray):
        hidden_states = torch.from_numpy(hidden_states).half()  # float16
    
    # Promediar atención entre cabezas
    avg_attention = attention.mean(dim=0)  # [seq_len, seq_len]
    
    # Crear edge_index y edge_attr
    edge_index = []
    edge_attr = []
    
    seq_len = avg_attention.shape[0]
    for i in range(seq_len):
        for j in range(seq_len):
            # IMPORTANTE: Solo permitir j <= i (causalidad: token i solo atiende al pasado)
            if j <= i:
                attn_value = avg_attention[i, j].item()
                if attn_value > attn_threshold:
                    # i presta atención a j: arco de j -> i
                    edge_index.append([j, i])
                    edge_attr.append(attn_value)
    
    if len(edge_index) == 0:
        # Grafo sin arcos
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0,), dtype=torch.half)  # float16
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.half)  # float16
    
    # Crear grafo PyG
    graph = Data(
        x=hidden_states,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=seq_len
    )
    
    return graph


def preprocess_batch(batch_file, labels_dict, args):
    """
    Pre-procesa un batch y genera 3 versiones:
    - LSTM-solo: hidden states del último token por capa
    - GNN-det+LSTM y GVAE: grafos con atenciones promediadas
    """
    # Cargar batch
    traces = load_trace_file(batch_file)
    
    # Extraer número de batch
    batch_num = extract_batch_number(batch_file)
    
    # Estructuras para cada modelo
    lstm_sequences = []
    gnn_graphs = []  # Para GNN-det+LSTM y GVAE (misma estructura)
    labels = []
    question_ids = []
    
    for trace in traces:
        question_id = trace['question_id']
        num_layers = len(trace['hidden_states'])
        
        # 1. LSTM-solo: hidden state del último token por capa
        import numpy as np
        lstm_seq = []
        for layer_idx in range(num_layers):
            last_token_hidden = trace['hidden_states'][layer_idx][-1]  # [hidden_dim]
            # Convertir a tensor si es numpy array (usar float16 para consistencia)
            if isinstance(last_token_hidden, np.ndarray):
                last_token_hidden = torch.from_numpy(last_token_hidden).half()  # float16
            lstm_seq.append(last_token_hidden)
        lstm_seq = torch.stack(lstm_seq, dim=0)  # [num_layers, hidden_dim]
        lstm_sequences.append(lstm_seq)
        
        # 2. GNN-det+LSTM y GVAE: grafos con atenciones promediadas por capa
        graphs_by_layer = []
        for layer_idx in range(num_layers):
            graph = build_graph_from_trace(trace, layer_idx, args.attn_threshold)
            graphs_by_layer.append(graph)
        gnn_graphs.append(graphs_by_layer)
        
        # Label
        label = labels_dict.get(question_id, 0)
        labels.append(label)
        question_ids.append(question_id)
    
    # Convertir LSTM sequences a tensor
    lstm_sequences = torch.stack(lstm_sequences, dim=0)  # [batch_size, num_layers, hidden_dim]
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    
    return {
        'lstm_solo': {
            'sequences': lstm_sequences,
            'labels': labels_tensor,
            'question_ids': question_ids
        },
        'gnn': {  # Para GNN-det+LSTM y GVAE
            'graphs': gnn_graphs,
            'labels': labels_tensor,
            'question_ids': question_ids
        },
        'batch_num': batch_num
    }


def preprocess_dataset(args):
    """
    Pre-procesa el dataset por batches y guarda archivos separados.
    """
    print("="*80)
    print("PRE-PROCESAMIENTO DE DATASET PARA ENTRENAMIENTO RÁPIDO")
    print("="*80)
    
    # 1. Encontrar archivos
    print("\n1. Encontrando archivos...")
    batch_files = sorted(glob.glob(args.data_pattern))
    print(f"   - {len(batch_files)} archivos encontrados")
    
    # 2. Cargar scores y crear etiquetas binarias
    print("\n2. Cargando scores y creando etiquetas...")
    scores_df = pd.read_csv(args.scores_file)
    scores_dict = dict(zip(scores_df['question_id'], scores_df['bleurt_score']))
    
    labels_dict = {}
    for qid, score in scores_dict.items():
        labels_dict[qid] = 1 if score < args.score_threshold else 0
    
    num_hallucinations = sum(labels_dict.values())
    num_correct = len(labels_dict) - num_hallucinations
    
    print(f"   - {len(labels_dict)} scores cargados")
    print(f"   - Balance: {num_hallucinations} alucinaciones ({100*num_hallucinations/len(labels_dict):.1f}%), "
          f"{num_correct} correctas ({100*num_correct/len(labels_dict):.1f}%)")
    
    # 3. Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    lstm_dir = output_dir / 'lstm_solo'
    gnn_dir = output_dir / 'gnn'  # Para GNN-det+LSTM y GVAE
    
    lstm_dir.mkdir(exist_ok=True)
    gnn_dir.mkdir(exist_ok=True)
    
    print(f"\n3. Procesando batches...")
    print(f"   Directorio de salida: {output_dir}")
    
    total_lstm_size = 0
    total_gnn_size = 0
    total_traces = 0
    
    # 4. Procesar cada batch
    for batch_file in tqdm(batch_files, desc="Procesando batches"):
        try:
            processed = preprocess_batch(batch_file, labels_dict, args)
            batch_num = processed['batch_num']
            
            if batch_num is None:
                batch_num = batch_files.index(batch_file)
            
            total_traces += len(processed['lstm_solo']['question_ids'])
            
            # Guardar LSTM-solo
            lstm_output = lstm_dir / f'batch_{batch_num:04d}.pt'
            torch.save(processed['lstm_solo'], lstm_output)
            total_lstm_size += lstm_output.stat().st_size
            
            # Guardar GNN (para GNN-det+LSTM y GVAE)
            gnn_output = gnn_dir / f'batch_{batch_num:04d}.pt'
            torch.save(processed['gnn'], gnn_output)
            total_gnn_size += gnn_output.stat().st_size
            
            # Liberar memoria
            del processed
            gc.collect()
            
        except Exception as e:
            print(f"\n   ⚠️  Error procesando {batch_file}: {e}")
            continue
    
    # 5. Estadísticas finales
    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)
    print(f"Batches procesados: {len(batch_files)}")
    print(f"Traces totales: {total_traces}")
    print(f"\nArchivos generados:")
    print(f"  LSTM-solo:")
    print(f"    - Directorio: {lstm_dir}")
    print(f"    - Tamaño total: {total_lstm_size/(1024**2):.2f} MB")
    print(f"  GNN-det+LSTM / GVAE:")
    print(f"    - Directorio: {gnn_dir}")
    print(f"    - Tamaño total: {total_gnn_size/(1024**2):.2f} MB")
    print(f"\nNOTA: GNN-det+LSTM y GVAE comparten la misma estructura de grafos")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pre-procesa dataset para entrenamiento rápido"
    )
    
    parser.add_argument(
        '--data-pattern',
        type=str,
        required=True,
        help='Patrón glob para archivos .pkl o .pkl.gz'
    )
    
    parser.add_argument(
        '--scores-file',
        type=str,
        required=True,
        help='Archivo CSV con scores BLEURT'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='preprocessed_data',
        help='Directorio de salida (default: preprocessed_data)'
    )
    
    parser.add_argument(
        '--attn-threshold',
        type=float,
        default=0.0,
        help='Umbral de atención para crear arcos (default: 0.0)'
    )
    
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.5,
        help='Umbral de score BLEURT para etiquetar alucinaciones (default: 0.5)'
    )
    
    args = parser.parse_args()
    preprocess_dataset(args)
