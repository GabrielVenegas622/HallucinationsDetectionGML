"""
Script de pre-procesamiento para aceler training.

Este script:
1. Carga TODOS los traces desde archivos .pkl/.pkl.gz
2. Genera los grafos PyG para cada capa
3. Extrae las representaciones layer_sequence (tensor compacto)
4. Guarda en formato .pt optimizado para carga rápida

Ventajas:
- Reduce tiempo de carga de ~30 seg a <1 seg por batch
- Archivo resultante es ~10-20× más pequeño
- Una vez procesado, el entrenamiento es 50-100× más rápido

Uso:
    python preprocess_for_training.py \
        --data-pattern "traces_data/*.pkl*" \
        --scores-file ground_truth_scores.csv \
        --output preprocessed_data.pt \
        --attn-threshold 0.0 \
        --score-threshold 0.5
"""

import torch
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc
from torch_geometric.nn import global_mean_pool
from dataloader import TraceGraphDataset


def preprocess_dataset(args):
    """
    Pre-procesa el dataset completo y guarda tensores optimizados.
    """
    print("="*80)
    print("PRE-PROCESAMIENTO DE DATASET PARA ENTRENAMIENTO RÁPIDO")
    print("="*80)
    
    # 1. Cargar dataset de grafos
    print("\n1. Cargando dataset de grafos...")
    graph_dataset = TraceGraphDataset(
        args.data_pattern,
        attn_threshold=args.attn_threshold,
        lazy_loading=False  # Cargar todo en memoria para pre-procesamiento
    )
    
    num_layers = graph_dataset.num_layers
    total_graphs = len(graph_dataset)
    num_traces = total_graphs // num_layers
    
    print(f"   - {num_traces} traces")
    print(f"   - {num_layers} capas por trace")
    print(f"   - {total_graphs} grafos totales")
    
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
    
    # 3. Extraer layer_sequences para cada trace
    print("\n3. Extrayendo layer sequences...")
    print("   Esto puede tomar varios minutos dependiendo del tamaño del dataset...")
    
    layer_sequences = []
    labels = []
    question_ids = []
    
    for trace_idx in tqdm(range(num_traces), desc="Procesando traces"):
        # Obtener todos los grafos de este trace
        graphs_by_layer = []
        question_id = None
        
        for layer_idx in range(num_layers):
            global_idx = trace_idx * num_layers + layer_idx
            graph = graph_dataset[global_idx]
            graphs_by_layer.append(graph)
            
            if question_id is None:
                question_id = graph.question_id
        
        # Extraer layer sequence
        layer_sequence = []
        for graph in graphs_by_layer:
            # Simular batch de 1 elemento
            batch = torch.zeros(graph.num_nodes, dtype=torch.long)
            layer_repr = global_mean_pool(graph.x, batch)  # [1, hidden_dim]
            layer_sequence.append(layer_repr.squeeze(0))  # [hidden_dim]
        
        layer_sequence = torch.stack(layer_sequence, dim=0)  # [num_layers, hidden_dim]
        
        # Obtener label
        label = labels_dict.get(question_id, 0)
        
        layer_sequences.append(layer_sequence)
        labels.append(label)
        question_ids.append(question_id)
        
        # Liberar memoria cada 100 traces
        if (trace_idx + 1) % 100 == 0:
            gc.collect()
    
    # 4. Convertir a tensores
    print("\n4. Convirtiendo a tensores...")
    layer_sequences_tensor = torch.stack(layer_sequences, dim=0)  # [num_traces, num_layers, hidden_dim]
    labels_tensor = torch.tensor(labels, dtype=torch.float)  # [num_traces]
    
    print(f"   - Layer sequences shape: {layer_sequences_tensor.shape}")
    print(f"   - Labels shape: {labels_tensor.shape}")
    
    # 5. Guardar en formato .pt
    print("\n5. Guardando datos pre-procesados...")
    output_path = Path(args.output)
    
    data_dict = {
        'layer_sequences': layer_sequences_tensor,
        'labels': labels_tensor,
        'question_ids': question_ids,
        'num_layers': num_layers,
        'hidden_dim': layer_sequences_tensor.shape[2],
        'score_threshold': args.score_threshold,
        'attn_threshold': args.attn_threshold
    }
    
    torch.save(data_dict, output_path)
    
    # Calcular tamaño del archivo
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"   ✅ Guardado en: {output_path}")
    print(f"   ✅ Tamaño: {file_size_mb:.2f} MB")
    
    # 6. Estadísticas finales
    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)
    print(f"Traces procesados: {num_traces}")
    print(f"Capas por trace: {num_layers}")
    print(f"Dimensión hidden: {layer_sequences_tensor.shape[2]}")
    print(f"Tamaño del archivo: {file_size_mb:.2f} MB")
    print(f"\nPara usar en entrenamiento:")
    print(f"  data = torch.load('{output_path}')")
    print(f"  layer_sequences = data['layer_sequences']  # Shape: {list(layer_sequences_tensor.shape)}")
    print(f"  labels = data['labels']  # Shape: {list(labels_tensor.shape)}")
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
        '--output',
        type=str,
        default='preprocessed_data.pt',
        help='Archivo de salida .pt (default: preprocessed_data.pt)'
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
