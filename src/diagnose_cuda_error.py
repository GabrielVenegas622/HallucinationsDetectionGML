#!/usr/bin/env python3
"""
Diagnóstico de Error CUDA CUBLAS_STATUS_EXECUTION_FAILED

Este script ayuda a diagnosticar y resolver errores de CUDA en GNN-det+LSTM.

Uso:
    python diagnose_cuda_error.py --data-pattern "traces_data/*.pkl" --scores-file scores.csv
"""

import torch
import argparse
import numpy as np
from baseline import SequentialTraceDataset, collate_sequential_batch, GNNDetLSTM
from torch.utils.data import DataLoader, Subset

def check_cuda_environment():
    """Verificar el ambiente CUDA"""
    print("="*60)
    print("1. VERIFICACIÓN DE AMBIENTE CUDA")
    print("="*60)
    
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Número de GPUs: {torch.cuda.device_count()}")
        print(f"GPU actual: {torch.cuda.current_device()}")
        print(f"Nombre GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Memoria asignada: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        print(f"Memoria reservada: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
    print()

def check_dataset_values(dataset, num_samples=10):
    """Verificar que los datos del dataset no tengan valores problemáticos"""
    print("="*60)
    print("2. VERIFICACIÓN DE DATOS DEL DATASET")
    print("="*60)
    
    issues_found = False
    
    for i in range(min(num_samples, len(dataset))):
        graphs, label, qid = dataset[i]
        
        for layer_idx, graph in enumerate(graphs):
            # Verificar node features
            if torch.isnan(graph.x).any():
                print(f"❌ NaN encontrado en trace {i}, capa {layer_idx}, x (features)")
                issues_found = True
            if torch.isinf(graph.x).any():
                print(f"❌ Inf encontrado en trace {i}, capa {layer_idx}, x (features)")
                issues_found = True
            
            # Verificar edge_attr
            if graph.edge_attr is not None and graph.edge_attr.numel() > 0:
                if torch.isnan(graph.edge_attr).any():
                    print(f"❌ NaN encontrado en trace {i}, capa {layer_idx}, edge_attr")
                    issues_found = True
                if torch.isinf(graph.edge_attr).any():
                    print(f"❌ Inf encontrado en trace {i}, capa {layer_idx}, edge_attr")
                    issues_found = True
                
                # Verificar valores fuera de rango [0, 1]
                if graph.edge_attr.max() > 1.0:
                    print(f"⚠️  edge_attr > 1.0 en trace {i}, capa {layer_idx}: max={graph.edge_attr.max().item()}")
                if graph.edge_attr.min() < 0.0:
                    print(f"⚠️  edge_attr < 0.0 en trace {i}, capa {layer_idx}: min={graph.edge_attr.min().item()}")
            
            # Verificar rangos de valores en x
            x_max = graph.x.max().item()
            x_min = graph.x.min().item()
            if abs(x_max) > 1e6 or abs(x_min) > 1e6:
                print(f"⚠️  Valores extremos en x: trace {i}, capa {layer_idx}, "
                      f"rango=[{x_min:.2e}, {x_max:.2e}]")
    
    if not issues_found:
        print("✓ No se encontraron NaN, Inf o valores extremos en los datos")
    print()
    
    return not issues_found

def test_gnn_on_cpu(dataset):
    """Probar GNN en CPU para descartar problemas de datos"""
    print("="*60)
    print("3. TEST DE GNN EN CPU")
    print("="*60)
    
    try:
        # Crear subset muy pequeño
        subset = Subset(dataset, range(min(4, len(dataset))))
        loader = DataLoader(subset, batch_size=2, collate_fn=collate_sequential_batch)
        
        # Obtener dimensiones
        sample_graph = dataset[0][0][0]
        hidden_dim = sample_graph.x.shape[1]
        
        # Crear modelo en CPU
        model = GNNDetLSTM(
            hidden_dim=hidden_dim,
            gnn_hidden=64,
            lstm_hidden=128,
            num_lstm_layers=2,
            dropout=0.3
        )
        model.eval()
        
        print(f"Modelo creado en CPU")
        print(f"Hidden dim: {hidden_dim}")
        
        # Forward pass en CPU
        with torch.no_grad():
            for batched_by_layer, labels, _ in loader:
                logits = model(batched_by_layer, len(batched_by_layer))
                print(f"✓ Forward pass en CPU exitoso")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                
                if torch.isnan(logits).any():
                    print(f"❌ NaN en logits!")
                    return False
                if torch.isinf(logits).any():
                    print(f"❌ Inf en logits!")
                    return False
                
                break
        
        print("✓ Test en CPU completado sin errores")
        print()
        return True
        
    except Exception as e:
        print(f"❌ Error en test de CPU: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def test_gnn_on_gpu_small_batch(dataset):
    """Probar GNN en GPU con batch pequeño"""
    print("="*60)
    print("4. TEST DE GNN EN GPU (BATCH PEQUEÑO)")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA no disponible, saltando test de GPU")
        print()
        return True
    
    try:
        device = torch.device('cuda')
        
        # Limpiar cache de CUDA
        torch.cuda.empty_cache()
        print("✓ Cache de CUDA limpiado")
        
        # Crear subset muy pequeño
        subset = Subset(dataset, range(min(4, len(dataset))))
        loader = DataLoader(subset, batch_size=1, collate_fn=collate_sequential_batch)
        
        # Obtener dimensiones
        sample_graph = dataset[0][0][0]
        hidden_dim = sample_graph.x.shape[1]
        
        # Crear modelo en GPU
        model = GNNDetLSTM(
            hidden_dim=hidden_dim,
            gnn_hidden=64,
            lstm_hidden=128,
            num_lstm_layers=2,
            dropout=0.3
        ).to(device)
        model.eval()
        
        print(f"Modelo creado en GPU")
        
        # Forward pass en GPU con batch_size=1
        with torch.no_grad():
            for batched_by_layer, labels, _ in loader:
                # Mover a GPU
                batched_by_layer = [data.to(device) for data in batched_by_layer]
                labels = labels.to(device)
                
                print(f"  Procesando batch en GPU...")
                logits = model(batched_by_layer, len(batched_by_layer))
                
                print(f"✓ Forward pass en GPU exitoso (batch_size=1)")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                
                if torch.isnan(logits).any():
                    print(f"❌ NaN en logits!")
                    return False
                if torch.isinf(logits).any():
                    print(f"❌ Inf en logits!")
                    return False
                
                break
        
        print("✓ Test en GPU completado sin errores")
        print()
        return True
        
    except RuntimeError as e:
        print(f"❌ Error en test de GPU: {e}")
        if "CUDA" in str(e) or "CUBLAS" in str(e):
            print("\n💡 RECOMENDACIONES:")
            print("   1. Ejecutar en CPU: Agregar el flag --cpu al entrenar")
            print("   2. Actualizar drivers de GPU")
            print("   3. Reducir batch_size a 4 u 8")
            print("   4. Verificar que edge_attr no tenga valores corruptos")
        print()
        return False

def provide_recommendations(cpu_works, gpu_works, data_clean):
    """Proveer recomendaciones basadas en los resultados"""
    print("="*60)
    print("5. RECOMENDACIONES")
    print("="*60)
    
    if data_clean and cpu_works and gpu_works:
        print("✅ TODO FUNCIONA CORRECTAMENTE")
        print("   Puedes proceder con el entrenamiento normal:")
        print()
        print("   python src/baseline.py \\")
        print("       --data-pattern 'traces_data/*.pkl' \\")
        print("       --scores-file scores.csv \\")
        print("       --batch-size 16 \\")
        print("       --epochs 50")
        
    elif data_clean and cpu_works and not gpu_works:
        print("⚠️  EL MODELO FUNCIONA EN CPU PERO NO EN GPU")
        print()
        print("   Soluciones posibles:")
        print()
        print("   OPCIÓN 1: Entrenar en CPU (más lento pero funcional)")
        print("   python src/baseline.py \\")
        print("       --data-pattern 'traces_data/*.pkl' \\")
        print("       --scores-file scores.csv \\")
        print("       --batch-size 8 \\")
        print("       --epochs 50")
        print("   # Agrega esto al código: device = torch.device('cpu')")
        print()
        print("   OPCIÓN 2: Solucionar problema de GPU")
        print("   - Actualizar drivers de NVIDIA")
        print("   - Actualizar PyTorch: pip install --upgrade torch torchvision")
        print("   - Reducir batch_size a 4 u 8")
        print("   - Limpiar cache: torch.cuda.empty_cache()")
        print()
        print("   OPCIÓN 3: Usar GNN más simple")
        print("   - Cambiar de GINEConv a GCNConv (no usa edge_attr)")
        
    elif not data_clean:
        print("❌ HAY PROBLEMAS EN LOS DATOS")
        print()
        print("   Los datos contienen NaN, Inf o valores extremos.")
        print("   Debes limpiar los datos antes de entrenar:")
        print()
        print("   1. Revisar el proceso de extracción de hidden_states")
        print("   2. Normalizar valores de atención (edge_attr)")
        print("   3. Aplicar clipping a valores extremos")
        
    else:
        print("❌ HAY PROBLEMAS EN EL MODELO")
        print()
        print("   El modelo falla incluso en CPU.")
        print("   Contactar soporte con el traceback completo.")
    
    print("="*60)

def main(args):
    print("\n" + "="*60)
    print("DIAGNÓSTICO DE ERROR CUDA CUBLAS")
    print("="*60)
    print()
    
    # 1. Check CUDA
    check_cuda_environment()
    
    # 2. Load dataset
    print("Cargando dataset...")
    dataset = SequentialTraceDataset(
        args.data_pattern,
        args.scores_file,
        attn_threshold=0.01,
        score_threshold=0.5
    )
    print(f"✓ Dataset cargado: {len(dataset)} traces\n")
    
    # 3. Check data
    data_clean = check_dataset_values(dataset, num_samples=args.num_samples)
    
    # 4. Test CPU
    cpu_works = test_gnn_on_cpu(dataset)
    
    # 5. Test GPU
    gpu_works = test_gnn_on_gpu_small_batch(dataset)
    
    # 6. Recommendations
    provide_recommendations(cpu_works, gpu_works, data_clean)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diagnóstico de error CUDA")
    
    parser.add_argument('--data-pattern', type=str, required=True,
                       help='Patrón glob para archivos .pkl')
    parser.add_argument('--scores-file', type=str, required=True,
                       help='Archivo CSV con scores BLEURT')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Número de muestras a verificar')
    
    args = parser.parse_args()
    
    main(args)
