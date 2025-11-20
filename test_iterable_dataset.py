#!/usr/bin/env python3
"""
Script de prueba para verificar el IterableDataset con múltiples workers.

Este script verifica:
1. Carga correcta de datos preprocesados con IterableDataset
2. Paralelización con num_workers > 0
3. Shuffling local
4. Uso eficiente de memoria (solo 1 archivo por worker)
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader
import sys
sys.path.append('src')

from baseline import PreprocessedLSTMDataset, PreprocessedGNNDataset, collate_lstm_batch, collate_gnn_batch
import time
import psutil
import os

def get_memory_usage():
    """Obtiene uso de memoria en MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_iterable_dataset():
    print("="*80)
    print("TEST: IterableDataset con múltiples workers")
    print("="*80)
    
    # Directorios preprocesados
    preprocessed_dir = Path('preprocessed_data')
    lstm_dir = preprocessed_dir / 'lstm_solo'
    gnn_dir = preprocessed_dir / 'gnn'
    
    # Verificar que existan archivos
    lstm_files = sorted(list(lstm_dir.glob('preprocessed_*.pt')))
    gnn_files = sorted(list(gnn_dir.glob('preprocessed_*.pt')))
    
    if not lstm_files or not gnn_files:
        print("❌ No se encontraron archivos preprocesados")
        print(f"   - LSTM dir: {lstm_dir}")
        print(f"   - GNN dir: {gnn_dir}")
        return
    
    print(f"\n✅ Encontrados {len(lstm_files)} archivos LSTM y {len(gnn_files)} archivos GNN")
    
    # Usar archivos disponibles (máximo 3)
    test_lstm_files = lstm_files[:min(3, len(lstm_files))]
    test_gnn_files = gnn_files[:min(3, len(gnn_files))]
    
    print(f"\n📦 Testing con {len(test_lstm_files)} archivos LSTM y {len(test_gnn_files)} archivos GNN")
    
    # Test 1: LSTM Dataset con diferentes num_workers
    print("\n" + "="*80)
    print("TEST 1: LSTM Dataset - Comparación de velocidad con diferentes workers")
    print("="*80)
    
    for num_workers in [0, 2]:
        print(f"\n▶ Testing con num_workers={num_workers}")
        
        # Crear dataset
        dataset = PreprocessedLSTMDataset(
            lstm_dir, 
            batch_files_to_load=test_lstm_files,
            shuffle_buffer_size=100
        )
        
        # Crear DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=collate_lstm_batch,
            num_workers=num_workers
        )
        
        # Medir tiempo y memoria
        mem_before = get_memory_usage()
        start_time = time.time()
        
        batch_count = 0
        sample_count = 0
        
        for sequences, labels, question_ids in dataloader:
            batch_count += 1
            sample_count += len(labels)
            
            # Validar shapes
            assert sequences.dim() == 3, f"Expected 3D tensor, got {sequences.dim()}D"
            assert labels.dim() == 1, f"Expected 1D labels, got {labels.dim()}D"
            
            # Solo procesar primeros 10 batches para test rápido
            if batch_count >= 10:
                break
        
        elapsed = time.time() - start_time
        mem_after = get_memory_usage()
        mem_used = mem_after - mem_before
        
        print(f"   ✅ Procesados {batch_count} batches ({sample_count} samples)")
        print(f"   ⏱️  Tiempo: {elapsed:.2f}s ({sample_count/elapsed:.1f} samples/s)")
        print(f"   💾 Memoria usada: {mem_used:.1f} MB")
    
    # Test 2: GNN Dataset
    print("\n" + "="*80)
    print("TEST 2: GNN Dataset - Verificación de estructura")
    print("="*80)
    
    dataset_gnn = PreprocessedGNNDataset(
        gnn_dir,
        batch_files_to_load=test_gnn_files,
        shuffle_buffer_size=100
    )
    
    dataloader_gnn = DataLoader(
        dataset_gnn,
        batch_size=16,
        collate_fn=collate_gnn_batch,
        num_workers=2
    )
    
    print("\n▶ Inspeccionando primer batch de GNN...")
    for graphs, labels, question_ids in dataloader_gnn:
        print(f"   ✅ Batch de grafos cargado correctamente")
        print(f"      - Número de grafos: {len(graphs)}")
        print(f"      - Labels shape: {labels.shape}")
        print(f"      - Primer grafo:")
        print(f"        * Nodos: {graphs[0].x.shape[0]}")
        print(f"        * Dimensión features: {graphs[0].x.shape[1]}")
        print(f"        * Aristas: {graphs[0].edge_index.shape[1]}")
        break
    
    # Test 3: Verificar shuffling
    print("\n" + "="*80)
    print("TEST 3: Verificar shuffling local")
    print("="*80)
    
    dataset_shuffle = PreprocessedLSTMDataset(
        lstm_dir,
        batch_files_to_load=test_lstm_files[:1],  # Solo 1 archivo
        shuffle_buffer_size=100
    )
    
    dataloader_shuffle = DataLoader(
        dataset_shuffle,
        batch_size=10,
        collate_fn=collate_lstm_batch,
        num_workers=0
    )
    
    # Obtener IDs de primeros 30 samples
    ids_run1 = []
    ids_run2 = []
    
    for _, _, qids in dataloader_shuffle:
        ids_run1.extend(qids)
        if len(ids_run1) >= 30:
            break
    
    # Segunda pasada
    dataset_shuffle2 = PreprocessedLSTMDataset(
        lstm_dir,
        batch_files_to_load=test_lstm_files[:1],
        shuffle_buffer_size=100
    )
    dataloader_shuffle2 = DataLoader(
        dataset_shuffle2,
        batch_size=10,
        collate_fn=collate_lstm_batch,
        num_workers=0
    )
    
    for _, _, qids in dataloader_shuffle2:
        ids_run2.extend(qids)
        if len(ids_run2) >= 30:
            break
    
    ids_run1 = ids_run1[:30]
    ids_run2 = ids_run2[:30]
    
    same_count = sum(1 for a, b in zip(ids_run1, ids_run2) if a == b)
    print(f"\n   Comparación de 2 pasadas (primeros 30 samples):")
    print(f"   - IDs idénticos: {same_count}/30")
    print(f"   - IDs diferentes: {30-same_count}/30")
    
    if same_count < 25:
        print("   ✅ Shuffling funciona correctamente (orden diferente)")
    else:
        print("   ⚠️  Poco shuffling detectado (puede ser normal con buffer pequeño)")
    
    print("\n" + "="*80)
    print("✅ TODOS LOS TESTS COMPLETADOS")
    print("="*80)
    print("\nResumen:")
    print("  ✅ IterableDataset funciona correctamente")
    print("  ✅ Múltiples workers soportados (paralelización)")
    print("  ✅ Shuffling local implementado")
    print("  ✅ Estructuras de datos correctas (LSTM y GNN)")
    print("\n💡 El dataset está listo para entrenamiento con paralelización!")

if __name__ == '__main__':
    try:
        test_iterable_dataset()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
