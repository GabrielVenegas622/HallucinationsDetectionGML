"""
Script de prueba para baseline.py

Este script realiza pruebas rápidas de todos los modelos con un subset pequeño
del dataset para verificar que no hay errores antes de entrenar completamente.

Uso:
    python test_baseline.py --data-pattern "traces_data/*.pkl" --scores-file ground_truth_scores.csv
"""

import torch
import torch.nn as nn
from torch.utils.data import Subset
import argparse
import sys
from pathlib import Path

# Importar desde baseline.py
from baseline import (
    LSTMBaseline,
    GNNDetLSTM,
    GVAELSTM,
    SequentialTraceDataset,
    collate_sequential_batch,
    train_lstm_baseline,
    train_gnn_det_lstm,
    train_gvae_lstm
)


def test_dataset_loading(args):
    """Test 1: Cargar el dataset y verificar que funciona"""
    print("\n" + "="*80)
    print("TEST 1: Cargando Dataset")
    print("="*80)
    
    try:
        dataset = SequentialTraceDataset(
            args.data_pattern,
            args.scores_file,
            attn_threshold=args.attn_threshold,
            score_threshold=args.score_threshold
        )
        
        print(f"✓ Dataset cargado exitosamente")
        print(f"  Total de traces: {len(dataset)}")
        print(f"  Capas por trace: {dataset.num_layers}")
        
        # Obtener una muestra
        sample = dataset[0]
        graphs, label, qid = sample
        print(f"  Muestra obtenida:")
        print(f"    - Número de grafos (capas): {len(graphs)}")
        print(f"    - Etiqueta: {label.item()}")
        print(f"    - Question ID: {qid}")
        print(f"    - Forma de features del primer grafo: {graphs[0].x.shape}")
        print(f"    - Número de nodos: {graphs[0].num_nodes}")
        print(f"    - Número de arcos: {graphs[0].edge_index.size(1)}")
        
        return dataset, True
    except Exception as e:
        print(f"✗ Error al cargar dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_dataloader(dataset, batch_size=4):
    """Test 2: Verificar el dataloader"""
    print("\n" + "="*80)
    print("TEST 2: Verificando DataLoader")
    print("="*80)
    
    try:
        from torch.utils.data import DataLoader
        
        # Crear subset pequeño
        subset_size = min(20, len(dataset))
        subset = Subset(dataset, range(subset_size))
        
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_sequential_batch
        )
        
        print(f"✓ DataLoader creado exitosamente")
        print(f"  Subset size: {subset_size}")
        print(f"  Batch size: {batch_size}")
        
        # Iterar sobre un batch
        for batched_by_layer, labels, question_ids in loader:
            print(f"\n  Primer batch cargado:")
            print(f"    - Número de capas: {len(batched_by_layer)}")
            print(f"    - Labels shape: {labels.shape}")
            print(f"    - Número de grafos en batch: {batched_by_layer[0].num_graphs}")
            print(f"    - Número total de nodos en capa 0: {batched_by_layer[0].num_nodes}")
            print(f"    - Número total de arcos en capa 0: {batched_by_layer[0].num_edges}")
            
            # Verificar edge_attr
            for i, layer_data in enumerate(batched_by_layer):
                if layer_data.edge_attr is not None:
                    edge_attr_shape = layer_data.edge_attr.shape
                    num_edges = layer_data.edge_index.size(1)
                    print(f"    - Capa {i}: edge_attr shape={edge_attr_shape}, num_edges={num_edges}")
                    if edge_attr_shape[0] != num_edges:
                        print(f"      ⚠️ WARNING: Mismatch entre edge_attr y edge_index!")
                        return False
            
            break
        
        return True
    except Exception as e:
        print(f"✗ Error en DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward(dataset, model_class, model_name, **model_kwargs):
    """Test 3: Verificar forward pass de un modelo"""
    print("\n" + "="*80)
    print(f"TEST 3: Forward Pass - {model_name}")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Dispositivo: {device}")
    
    try:
        # Crear modelo
        model = model_class(**model_kwargs).to(device)
        print(f"✓ Modelo {model_name} creado")
        print(f"  Parámetros: {sum(p.numel() for p in model.parameters()):,}")
        
        # Crear dataloader pequeño
        from torch.utils.data import DataLoader
        subset = Subset(dataset, range(min(8, len(dataset))))
        loader = DataLoader(subset, batch_size=4, collate_fn=collate_sequential_batch)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            for batched_by_layer, labels, _ in loader:
                labels = labels.to(device).unsqueeze(1)
                
                if model_name == "LSTM-solo":
                    # Extraer secuencia para LSTM
                    from torch_geometric.nn import global_mean_pool
                    layer_sequence = []
                    for layer_data in batched_by_layer:
                        layer_repr = global_mean_pool(
                            layer_data.x.to(device),
                            layer_data.batch.to(device)
                        )
                        layer_sequence.append(layer_repr)
                    layer_sequence = torch.stack(layer_sequence, dim=1)
                    
                    logits = model(layer_sequence)
                else:
                    # GNN models
                    batched_by_layer = [data.to(device) for data in batched_by_layer]
                    
                    if model_name == "GVAE+LSTM":
                        logits, _, _, _, _ = model(batched_by_layer, len(batched_by_layer))
                    else:
                        logits = model(batched_by_layer, len(batched_by_layer))
                
                print(f"✓ Forward pass exitoso")
                print(f"  Input: batch_size={labels.shape[0]}, num_layers={len(batched_by_layer)}")
                print(f"  Output logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                
                # Verificar que se puede calcular la loss
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, labels)
                print(f"  Loss calculada: {loss.item():.4f}")
                
                break
        
        return True
    except Exception as e:
        print(f"✗ Error en forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(dataset, model_class, model_name, train_fn, **model_kwargs):
    """Test 4: Verificar un paso de entrenamiento"""
    print("\n" + "="*80)
    print(f"TEST 4: Training Step - {model_name}")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Crear subset muy pequeño
        subset_size = min(20, len(dataset))
        train_subset = Subset(dataset, range(subset_size))
        val_subset = Subset(dataset, range(subset_size // 2))
        
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_subset, batch_size=4, collate_fn=collate_sequential_batch)
        val_loader = DataLoader(val_subset, batch_size=4, collate_fn=collate_sequential_batch)
        
        # Crear modelo
        model = model_class(**model_kwargs)
        
        # Entrenar por 2 épocas
        print(f"  Entrenando por 2 épocas...")
        if model_name == "GVAE+LSTM":
            history = train_fn(model, train_loader, val_loader, device, epochs=2, lr=0.001, kl_weight=0.001)
        else:
            history = train_fn(model, train_loader, val_loader, device, epochs=2, lr=0.001)
        
        print(f"✓ Entrenamiento completado")
        print(f"  Epoch 1 - Train Loss: {history['train_loss'][0]:.4f}, Val AUROC: {history['val_auroc'][0]:.4f}")
        print(f"  Epoch 2 - Train Loss: {history['train_loss'][1]:.4f}, Val AUROC: {history['val_auroc'][1]:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error en training step: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(args):
    """Ejecutar todos los tests"""
    print("="*80)
    print("INICIANDO TESTS DE BASELINE.PY")
    print("="*80)
    print(f"\nConfiguración:")
    print(f"  Data pattern: {args.data_pattern}")
    print(f"  Scores file: {args.scores_file}")
    print(f"  Score threshold: {args.score_threshold}")
    print(f"  Attn threshold: {args.attn_threshold}")
    
    results = {}
    
    # Test 1: Dataset
    dataset, success = test_dataset_loading(args)
    results['Dataset Loading'] = success
    if not success:
        print("\n❌ Tests abortados: No se pudo cargar el dataset")
        return results
    
    # Test 2: DataLoader
    success = test_dataloader(dataset, batch_size=args.batch_size)
    results['DataLoader'] = success
    if not success:
        print("\n⚠️ WARNING: Problemas con DataLoader, continuando...")
    
    # Obtener dimensiones para modelos
    sample_graph = dataset[0][0][0]
    hidden_dim = sample_graph.x.shape[1]
    print(f"\nDimensión de hidden states: {hidden_dim}")
    
    # Test 3: Forward Pass - LSTM
    success = test_model_forward(
        dataset,
        LSTMBaseline,
        "LSTM-solo",
        hidden_dim=hidden_dim,
        lstm_hidden=128,
        num_lstm_layers=2,
        dropout=0.3
    )
    results['LSTM Forward'] = success
    
    # Test 3: Forward Pass - GNN-det
    success = test_model_forward(
        dataset,
        GNNDetLSTM,
        "GNN-det+LSTM",
        hidden_dim=hidden_dim,
        gnn_hidden=64,
        lstm_hidden=128,
        num_lstm_layers=2,
        dropout=0.3
    )
    results['GNN-det Forward'] = success
    
    # Test 3: Forward Pass - GVAE
    success = test_model_forward(
        dataset,
        GVAELSTM,
        "GVAE+LSTM",
        hidden_dim=hidden_dim,
        gnn_hidden=64,
        latent_dim=32,
        lstm_hidden=128,
        num_lstm_layers=2,
        dropout=0.3
    )
    results['GVAE Forward'] = success
    
    # Test 4: Training Steps (solo si los forward funcionaron)
    if args.test_training:
        if results.get('LSTM Forward', False):
            success = test_training_step(
                dataset,
                LSTMBaseline,
                "LSTM-solo",
                train_lstm_baseline,
                hidden_dim=hidden_dim,
                lstm_hidden=128,
                num_lstm_layers=2,
                dropout=0.3
            )
            results['LSTM Training'] = success
        
        if results.get('GNN-det Forward', False):
            success = test_training_step(
                dataset,
                GNNDetLSTM,
                "GNN-det+LSTM",
                train_gnn_det_lstm,
                hidden_dim=hidden_dim,
                gnn_hidden=64,
                lstm_hidden=128,
                num_lstm_layers=2,
                dropout=0.3
            )
            results['GNN-det Training'] = success
        
        if results.get('GVAE Forward', False):
            success = test_training_step(
                dataset,
                GVAELSTM,
                "GVAE+LSTM",
                train_gvae_lstm,
                hidden_dim=hidden_dim,
                gnn_hidden=64,
                latent_dim=32,
                lstm_hidden=128,
                num_lstm_layers=2,
                dropout=0.3
            )
            results['GVAE Training'] = success
    
    return results


def print_results_summary(results):
    """Imprimir resumen de resultados"""
    print("\n" + "="*80)
    print("RESUMEN DE TESTS")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:<30} {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    if all_passed:
        print("✅ TODOS LOS TESTS PASARON - El script está listo para usar")
    else:
        print("❌ ALGUNOS TESTS FALLARON - Revisar los errores arriba")
    print("="*80)
    
    return all_passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script de prueba para baseline.py"
    )
    
    # Datos
    parser.add_argument('--data-pattern', type=str, required=True,
                       help='Patrón glob para archivos .pkl')
    parser.add_argument('--scores-file', type=str, required=True,
                       help='Archivo CSV con scores BLEURT')
    parser.add_argument('--attn-threshold', type=float, default=0.01,
                       help='Umbral de atención para crear arcos')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                       help='Umbral de score BLEURT para etiquetar alucinaciones')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Tamaño del batch para testing')
    parser.add_argument('--test-training', action='store_true',
                       help='Ejecutar también tests de entrenamiento (más lento)')
    
    args = parser.parse_args()
    
    # Ejecutar tests
    results = run_all_tests(args)
    
    # Imprimir resumen
    all_passed = print_results_summary(results)
    
    # Exit code
    sys.exit(0 if all_passed else 1)
