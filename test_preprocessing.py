"""
Script de prueba para verificar la carga de datos preprocesados.

Este script:
1. Asume que ya existen archivos preprocesados en preprocessed_data/
2. Verifica que los datasets pueden cargarlos sin errores
3. Imprime información de dimensiones y estadísticas
4. Prueba con DataLoader

Uso:
    python test_preprocessing.py --preprocessed-dir preprocessed_data
"""

import torch
import argparse
from pathlib import Path
import sys
import gc

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from baseline import PreprocessedLSTMDataset, PreprocessedGNNDataset


def test_preprocessing(args):
    """Prueba la carga de datos preprocesados"""
    
    print("="*80)
    print("TEST DE CARGA DE DATOS PREPROCESADOS")
    print("="*80)
    
    # Paso 1: Verificar archivos existentes
    print("\n📦 PASO 1: Verificando archivos preprocesados...")
    print("-"*80)
    
    test_dir = Path(args.preprocessed_dir)
    lstm_dir = test_dir / "lstm_solo"
    gnn_dir = test_dir / "gnn"
    
    if not lstm_dir.exists() or not gnn_dir.exists():
        print(f"❌ Directorios no encontrados: {lstm_dir} o {gnn_dir}")
        print(f"💡 Primero debes ejecutar preprocess_for_training.py")
        return False
    
    lstm_files = sorted(list(lstm_dir.glob('preprocessed_*.pt')))
    gnn_files = sorted(list(gnn_dir.glob('preprocessed_*.pt')))
    
    print(f"📂 Archivos LSTM encontrados: {len(lstm_files)}")
    print(f"📂 Archivos GNN encontrados: {len(gnn_files)}")
    
    if not lstm_files or not gnn_files:
        print("❌ No se encontraron archivos preprocesados")
        return False
    
    # Limitar a primer archivo para test rápido
    lstm_test_files = lstm_files[:1]
    gnn_test_files = gnn_files[:1]
    
    print(f"🔬 Probando con primer archivo de cada tipo...")
    
    # Paso 2: Cargar con datasets
    print("\n✅ PASO 2: Probando carga con PreprocessedLSTMDataset...")
    print("-"*80)
    
    try:
        lstm_dataset = PreprocessedLSTMDataset(
            lstm_dir, 
            batch_files_to_load=lstm_test_files,
            shuffle_buffer_size=0
        )
        print(f"✅ Dataset LSTM creado: {len(lstm_dataset)} traces")
        
        # Iterar sobre algunas muestras
        print("\n🔍 Iterando sobre primeras 5 muestras LSTM...")
        sample_count = 0
        for sequence, label, qid in lstm_dataset:
            sample_count += 1
            print(f"  Trace {sample_count}: sequence shape={sequence.shape}, label={label}, qid={qid}")
            if sample_count >= 5:
                break
        
        if sample_count == 0:
            print("❌ No se pudieron iterar muestras del dataset LSTM")
            return False
            
    except Exception as e:
        print(f"❌ Error al cargar dataset LSTM: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ PASO 3: Probando carga con PreprocessedGNNDataset...")
    print("-"*80)
    
    try:
        gnn_dataset = PreprocessedGNNDataset(
            gnn_dir,
            batch_files_to_load=gnn_test_files,
            shuffle_buffer_size=0
        )
        print(f"✅ Dataset GNN creado: {len(gnn_dataset)} traces")
        
        # Iterar sobre algunas muestras
        print("\n🔍 Iterando sobre primeras 5 muestras GNN...")
        sample_count = 0
        for graphs_list, label, qid in gnn_dataset:
            sample_count += 1
            print(f"  Trace {sample_count}: {len(graphs_list)} grafos, label={label}, qid={qid}")
            if len(graphs_list) > 0:
                print(f"    Grafo[0]: nodes={graphs_list[0].num_nodes}, edges={graphs_list[0].num_edges}, features={graphs_list[0].x.shape}")
            if sample_count >= 5:
                break
        
        if sample_count == 0:
            print("❌ No se pudieron iterar muestras del dataset GNN")
            return False
            
    except Exception as e:
        print(f"❌ Error al cargar dataset GNN: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Paso 4: Verificar tipos de datos
    print("\n✅ PASO 4: Verificando tipos de datos...")
    print("-"*80)
    
    try:
        # Cargar primer archivo LSTM
        lstm_file = lstm_test_files[0]
        batch_data = torch.load(lstm_file, weights_only=False)
        
        print(f"📊 Contenido del archivo LSTM:")
        print(f"  - Keys: {list(batch_data.keys())}")
        print(f"  - sequences: {len(batch_data['sequences'])} traces")
        print(f"  - labels: {len(batch_data['labels'])} etiquetas")
        print(f"  - question_ids: {len(batch_data['question_ids'])} IDs")
        
        # Verificar tipos
        sample_seq = batch_data['sequences'][0]
        print(f"\n🔍 Sample sequence:")
        print(f"  - Shape: {sample_seq.shape}")
        print(f"  - Dtype: {sample_seq.dtype}")
        print(f"  - Min/Max: {sample_seq.min():.4f} / {sample_seq.max():.4f}")
        
        del batch_data
        gc.collect()
        
        # Cargar primer archivo GNN
        gnn_file = gnn_test_files[0]
        batch_data = torch.load(gnn_file, weights_only=False)
        
        print(f"\n📊 Contenido del archivo GNN:")
        print(f"  - Keys: {list(batch_data.keys())}")
        print(f"  - graphs: {len(batch_data['graphs'])} traces")
        print(f"  - labels: {len(batch_data['labels'])} etiquetas")
        print(f"  - question_ids: {len(batch_data['question_ids'])} IDs")
        
        # Verificar tipos
        sample_graphs = batch_data['graphs'][0]
        print(f"\n🔍 Sample graphs (trace con {len(sample_graphs)} capas):")
        if len(sample_graphs) > 0:
            print(f"  - Grafo[0]: nodes={sample_graphs[0].num_nodes}, edges={sample_graphs[0].num_edges}")
            print(f"  - Node features shape: {sample_graphs[0].x.shape}")
            print(f"  - Node features dtype: {sample_graphs[0].x.dtype}")
            print(f"  - Edge attr dtype: {sample_graphs[0].edge_attr.dtype if sample_graphs[0].edge_attr is not None else 'None'}")
        
        del batch_data
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error al verificar tipos de datos: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Paso 5: Test con DataLoader
    print("\n✅ PASO 5: Probando con DataLoader (batch_size=4)...")
    print("-"*80)
    
    try:
        from torch.utils.data import DataLoader
        
        # Test LSTM con DataLoader
        lstm_dataset = PreprocessedLSTMDataset(lstm_dir, batch_files_to_load=lstm_test_files, shuffle_buffer_size=0)
        lstm_loader = DataLoader(lstm_dataset, batch_size=4, num_workers=0)
        
        print(f"🔄 Iterando sobre DataLoader LSTM (batch_size=4)...")
        batch_count = 0
        for batch in lstm_loader:
            sequences, labels, qids = batch
            batch_count += 1
            print(f"  Batch {batch_count}: sequences={sequences.shape}, labels={labels.shape}, dtype={sequences.dtype}")
            if batch_count >= 2:
                break
        
        # Test GNN con DataLoader
        gnn_dataset = PreprocessedGNNDataset(gnn_dir, batch_files_to_load=gnn_test_files, shuffle_buffer_size=0)
        
        # Para GNN necesitamos un collate_fn especial
        def collate_gnn(batch):
            graphs_list = [item[0] for item in batch]
            labels = torch.tensor([item[1] for item in batch])
            qids = [item[2] for item in batch]
            return graphs_list, labels, qids
        
        gnn_loader = DataLoader(gnn_dataset, batch_size=4, num_workers=0, collate_fn=collate_gnn)
        
        print(f"\n🔄 Iterando sobre DataLoader GNN (batch_size=4)...")
        batch_count = 0
        for batch in gnn_loader:
            graphs_list, labels, qids = batch
            batch_count += 1
            print(f"  Batch {batch_count}: {len(graphs_list)} traces, labels={labels.shape}")
            if len(graphs_list) > 0 and len(graphs_list[0]) > 0:
                print(f"    Primer trace: {len(graphs_list[0])} grafos")
                print(f"    Primer grafo dtype: x={graphs_list[0][0].x.dtype}, edge_attr={graphs_list[0][0].edge_attr.dtype if graphs_list[0][0].edge_attr is not None else 'None'}")
            if batch_count >= 2:
                break
        
    except Exception as e:
        print(f"❌ Error al probar DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
    print("="*80)
    print("\n💡 El dataset está correctamente preprocesado y listo para entrenamiento")
    print(f"   Directorios:")
    print(f"   - LSTM: {lstm_dir}")
    print(f"   - GNN: {gnn_dir}")
    print(f"\n💡 Para entrenar, usa:")
    print(f"   python src/baseline.py --preprocessed-dir {args.preprocessed_dir} --epochs 50")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de carga de datos preprocesados")
    parser.add_argument('--preprocessed-dir', type=str, default='preprocessed_data',
                       help='Directorio con datos preprocesados (default: preprocessed_data)')
    
    args = parser.parse_args()
    
    success = test_preprocessing(args)
    sys.exit(0 if success else 1)
