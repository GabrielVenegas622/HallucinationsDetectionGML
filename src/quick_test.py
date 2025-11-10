#!/usr/bin/env python3
"""
Quick Test - Verificación rápida de baseline.py

Este script ejecuta una verificación mínima para confirmar que todo funciona.
Usa solo 100 muestras y 1 época para ser muy rápido.

Uso:
    python quick_test.py --data-pattern "traces_data/*.pkl" --scores-file ground_truth_scores.csv
"""

import torch
import argparse
import sys
from pathlib import Path

print("Importando módulos...")
from baseline import (
    LSTMBaseline,
    GNNDetLSTM,
    GVAELSTM,
    SequentialTraceDataset,
    collate_sequential_batch
)
from torch.utils.data import DataLoader, Subset

def quick_test(args):
    """Test rápido de todos los componentes"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"QUICK TEST - baseline.py")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Data pattern: {args.data_pattern}")
    print(f"Scores file: {args.scores_file}")
    
    # 1. Cargar dataset
    print(f"\n[1/5] Cargando dataset...")
    try:
        dataset = SequentialTraceDataset(
            args.data_pattern,
            args.scores_file,
            attn_threshold=0.01,
            score_threshold=0.5
        )
        print(f"✓ Dataset cargado: {len(dataset)} traces")
    except Exception as e:
        print(f"✗ Error al cargar dataset: {e}")
        return False
    
    # 2. Crear subset pequeño
    num_samples = min(args.num_samples, len(dataset))
    print(f"\n[2/5] Creando subset de {num_samples} muestras...")
    subset = Subset(dataset, range(num_samples))
    
    train_size = int(0.7 * num_samples)
    val_size = num_samples - train_size
    
    train_subset = Subset(dataset, range(train_size))
    val_subset = Subset(dataset, range(train_size, num_samples))
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_sequential_batch)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_sequential_batch)
    
    print(f"✓ DataLoaders creados: train={train_size}, val={val_size}")
    
    # 3. Test forward pass de cada modelo
    print(f"\n[3/5] Testeando forward pass...")
    
    sample_graph = dataset[0][0][0]
    hidden_dim = sample_graph.x.shape[1]
    print(f"  Hidden dim: {hidden_dim}")
    
    models_to_test = []
    
    if args.test_lstm:
        models_to_test.append(('LSTM-solo', LSTMBaseline(
            hidden_dim=hidden_dim,
            lstm_hidden=128,
            num_lstm_layers=2,
            dropout=0.3
        )))
    
    if args.test_gnn:
        models_to_test.append(('GNN-det+LSTM', GNNDetLSTM(
            hidden_dim=hidden_dim,
            gnn_hidden=64,
            lstm_hidden=128,
            num_lstm_layers=2,
            dropout=0.3
        )))
    
    if args.test_gvae:
        models_to_test.append(('GVAE+LSTM', GVAELSTM(
            hidden_dim=hidden_dim,
            gnn_hidden=64,
            latent_dim=32,
            lstm_hidden=128,
            num_lstm_layers=2,
            dropout=0.3
        )))
    
    for model_name, model in models_to_test:
        try:
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                for batched_by_layer, labels, _ in train_loader:
                    labels = labels.to(device).unsqueeze(1)
                    
                    if model_name == "LSTM-solo":
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
                        batched_by_layer = [data.to(device) for data in batched_by_layer]
                        if model_name == "GVAE+LSTM":
                            logits, _, _, _, _ = model(batched_by_layer, len(batched_by_layer))
                        else:
                            logits = model(batched_by_layer, len(batched_by_layer))
                    
                    print(f"  ✓ {model_name}: logits shape={logits.shape}")
                    break
        except Exception as e:
            print(f"  ✗ {model_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 4. Test backward pass (1 batch)
    print(f"\n[4/5] Testeando backward pass...")
    try:
        criterion = torch.nn.BCEWithLogitsLoss()
        
        for model_name, model in models_to_test:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            for batched_by_layer, labels, _ in train_loader:
                labels = labels.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                
                if model_name == "LSTM-solo":
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
                    loss = criterion(logits, labels)
                else:
                    batched_by_layer = [data.to(device) for data in batched_by_layer]
                    if model_name == "GVAE+LSTM":
                        logits, mu_list, logvar_list, orig_list, recon_list = model(
                            batched_by_layer, len(batched_by_layer)
                        )
                        loss = criterion(logits, labels)
                        # Agregar VAE loss simplificado
                        for mu, logvar in zip(mu_list, logvar_list):
                            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                            loss = loss + 0.0001 * kld / len(mu_list)
                    else:
                        logits = model(batched_by_layer, len(batched_by_layer))
                        loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                print(f"  ✓ {model_name}: loss={loss.item():.4f}")
                break
    except Exception as e:
        print(f"  ✗ Backward pass ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test métricas
    print(f"\n[5/5] Testeando cálculo de métricas...")
    try:
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        
        model = models_to_test[0][1]  # Usar primer modelo
        model_name = models_to_test[0][0]
        model.eval()
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batched_by_layer, labels, _ in val_loader:
                labels_np = labels.numpy()
                labels = labels.to(device).unsqueeze(1)
                
                if model_name == "LSTM-solo":
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
                    batched_by_layer = [data.to(device) for data in batched_by_layer]
                    if model_name == "GVAE+LSTM":
                        logits, _, _, _, _ = model(batched_by_layer, len(batched_by_layer))
                    else:
                        logits = model(batched_by_layer, len(batched_by_layer))
                
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(labels_np)
        
        # Calcular métricas
        auroc = roc_auc_score(all_labels, all_probs)
        preds = [1 if p > 0.5 else 0 for p in all_probs]
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)
        
        print(f"  ✓ Métricas calculadas:")
        print(f"    AUROC: {auroc:.4f}")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    F1-Score: {f1:.4f}")
        
    except Exception as e:
        print(f"  ✗ Métricas ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quick test de baseline.py")
    
    parser.add_argument('--data-pattern', type=str, required=True,
                       help='Patrón glob para archivos .pkl')
    parser.add_argument('--scores-file', type=str, required=True,
                       help='Archivo CSV con scores BLEURT')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Número de muestras para el test')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Tamaño del batch')
    parser.add_argument('--test-lstm', action='store_true', default=True,
                       help='Testear LSTM-solo')
    parser.add_argument('--test-gnn', action='store_true', default=True,
                       help='Testear GNN-det+LSTM')
    parser.add_argument('--test-gvae', action='store_true', default=True,
                       help='Testear GVAE+LSTM')
    
    args = parser.parse_args()
    
    success = quick_test(args)
    
    print(f"\n{'='*60}")
    if success:
        print("✅ QUICK TEST PASADO - Todo funciona correctamente!")
        print("   Puedes proceder con el entrenamiento completo.")
    else:
        print("❌ QUICK TEST FALLIDO - Revisar errores arriba")
    print(f"{'='*60}\n")
    
    sys.exit(0 if success else 1)
