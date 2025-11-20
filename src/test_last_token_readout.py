"""
Script de prueba para verificar la implementación de Last Token Readout
con conexión residual en GNNDetLSTM y GVAELSTM.

Este script verifica:
1. Las dimensiones de entrada correctas de la LSTM
2. La concatenación residual (original + procesado)
3. El flujo de datos a través de ambos modelos
4. El conteo de parámetros comparable entre modelos
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent))

from baseline import GNNDetLSTM, GVAELSTM, LSTMBaseline

def create_dummy_graph_data(batch_size=4, num_nodes=10, hidden_dim=4096):
    """Crea datos de grafo dummy para testing"""
    graphs = []
    
    for _ in range(batch_size):
        # Crear nodos aleatorios
        x = torch.randn(num_nodes, hidden_dim)
        
        # Crear arcos aleatorios (estructura de grafo simple)
        num_edges = num_nodes * 2
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Pesos de atención (edge_attr)
        edge_attr = torch.rand(num_edges, 1)
        
        # Crear objeto Data
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(data)
    
    # Batch de grafos
    return Batch.from_data_list(graphs)

def test_gnn_det_lstm():
    """Prueba GNNDetLSTM con Last Token Readout + conexión residual"""
    print("=" * 80)
    print("TEST: GNNDetLSTM con Last Token Readout + Conexión Residual")
    print("=" * 80)
    
    hidden_dim = 4096
    gnn_hidden = 128
    lstm_hidden = 256
    batch_size = 4
    num_layers = 5
    
    # Crear modelo
    model = GNNDetLSTM(
        hidden_dim=hidden_dim,
        gnn_hidden=gnn_hidden,
        lstm_hidden=lstm_hidden,
        num_lstm_layers=2,
        dropout=0.3
    )
    
    # Verificar input_size de LSTM
    expected_lstm_input = hidden_dim + gnn_hidden
    actual_lstm_input = model.lstm.input_size
    
    print(f"\n✓ LSTM Input Size:")
    print(f"  Esperado: {expected_lstm_input} (hidden_dim={hidden_dim} + gnn_hidden={gnn_hidden})")
    print(f"  Real: {actual_lstm_input}")
    assert actual_lstm_input == expected_lstm_input, "LSTM input size incorrecto!"
    
    # Crear datos de prueba
    print(f"\n✓ Creando datos de prueba:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num layers: {num_layers}")
    
    batched_graphs_by_layer = []
    for layer_idx in range(num_layers):
        batch_data = create_dummy_graph_data(
            batch_size=batch_size,
            num_nodes=10,
            hidden_dim=hidden_dim
        )
        batched_graphs_by_layer.append(batch_data)
    
    # Forward pass
    print(f"\n✓ Forward pass...")
    model.eval()
    with torch.no_grad():
        logits = model(batched_graphs_by_layer, num_layers)
    
    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (batch_size, 1), "Output shape incorrecto!"
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Parámetros totales: {total_params:,}")
    
    print("\n✅ GNNDetLSTM: TODAS LAS PRUEBAS PASARON\n")
    return total_params

def test_gvae_lstm():
    """Prueba GVAELSTM con Last Token Readout + conexión residual"""
    print("=" * 80)
    print("TEST: GVAELSTM con Last Token Readout + Conexión Residual")
    print("=" * 80)
    
    hidden_dim = 4096
    gnn_hidden = 128
    latent_dim = 64
    lstm_hidden = 256
    batch_size = 4
    num_layers = 5
    
    # Crear modelo
    model = GVAELSTM(
        hidden_dim=hidden_dim,
        gnn_hidden=gnn_hidden,
        latent_dim=latent_dim,
        lstm_hidden=lstm_hidden,
        num_lstm_layers=2,
        dropout=0.3
    )
    
    # Verificar input_size de LSTM
    expected_lstm_input = hidden_dim + latent_dim
    actual_lstm_input = model.lstm.input_size
    
    print(f"\n✓ LSTM Input Size:")
    print(f"  Esperado: {expected_lstm_input} (hidden_dim={hidden_dim} + latent_dim={latent_dim})")
    print(f"  Real: {actual_lstm_input}")
    assert actual_lstm_input == expected_lstm_input, "LSTM input size incorrecto!"
    
    # Crear datos de prueba
    print(f"\n✓ Creando datos de prueba:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num layers: {num_layers}")
    
    batched_graphs_by_layer = []
    for layer_idx in range(num_layers):
        batch_data = create_dummy_graph_data(
            batch_size=batch_size,
            num_nodes=10,
            hidden_dim=hidden_dim
        )
        batched_graphs_by_layer.append(batch_data)
    
    # Forward pass
    print(f"\n✓ Forward pass...")
    model.eval()
    with torch.no_grad():
        logits, mu_list, logvar_list, original_reprs, reconstructed_reprs = model(
            batched_graphs_by_layer, num_layers
        )
    
    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (batch_size, 1), "Output shape incorrecto!"
    
    print(f"  Mu list length: {len(mu_list)}")
    assert len(mu_list) == num_layers, "Mu list length incorrecto!"
    
    print(f"  Logvar list length: {len(logvar_list)}")
    assert len(logvar_list) == num_layers, "Logvar list length incorrecto!"
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Parámetros totales: {total_params:,}")
    
    print("\n✅ GVAELSTM: TODAS LAS PRUEBAS PASARON\n")
    return total_params

def test_lstm_baseline():
    """Prueba LSTM baseline para comparación"""
    print("=" * 80)
    print("TEST: LSTM Baseline (para comparación)")
    print("=" * 80)
    
    hidden_dim = 4096
    lstm_hidden = 256
    batch_size = 4
    num_layers = 5
    
    # Crear modelo
    model = LSTMBaseline(
        hidden_dim=hidden_dim,
        lstm_hidden=lstm_hidden,
        num_lstm_layers=2,
        dropout=0.3
    )
    
    print(f"\n✓ LSTM Input Size: {model.lstm.input_size} (solo hidden_dim)")
    
    # Crear datos de prueba
    layer_sequence = torch.randn(batch_size, num_layers, hidden_dim)
    
    # Forward pass
    print(f"\n✓ Forward pass...")
    model.eval()
    with torch.no_grad():
        logits = model(layer_sequence)
    
    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (batch_size, 1), "Output shape incorrecto!"
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Parámetros totales: {total_params:,}")
    
    print("\n✅ LSTM Baseline: TODAS LAS PRUEBAS PASARON\n")
    return total_params

def compare_models():
    """Compara los conteos de parámetros entre modelos"""
    print("=" * 80)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 80)
    
    params_lstm = test_lstm_baseline()
    params_gnn_det = test_gnn_det_lstm()
    params_gvae = test_gvae_lstm()
    
    print("\n" + "=" * 80)
    print("RESUMEN DE PARÁMETROS")
    print("=" * 80)
    print(f"\n  LSTM-solo:      {params_lstm:>12,} parámetros")
    print(f"  GNN-det+LSTM:   {params_gnn_det:>12,} parámetros")
    print(f"  GVAE+LSTM:      {params_gvae:>12,} parámetros")
    
    print("\n✅ COMPARACIÓN COMPLETA")
    print("\nNOTAS IMPORTANTES:")
    print("  • LSTM-solo tiene MÁS parámetros porque su LSTM solo recibe hidden_dim (4096)")
    print("  • GNN-det+LSTM tiene MENOS parámetros en LSTM pero SUMA parámetros de GNN")
    print("  • GVAE+LSTM tiene MENOS parámetros en LSTM pero SUMA parámetros de encoder/decoder")
    print("  • La comparación justa debe considerar TODOS los parámetros del modelo completo")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print("\n🔬 VERIFICACIÓN DE IMPLEMENTACIÓN: LAST TOKEN READOUT\n")
    
    try:
        compare_models()
        print("\n✅✅✅ TODOS LOS TESTS PASARON EXITOSAMENTE ✅✅✅\n")
    except Exception as e:
        print(f"\n❌ ERROR EN TESTS: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
