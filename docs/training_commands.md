# Comandos de Entrenamiento - baseline.py

## Entrenamiento Completo

Entrena todos los modelos (LSTM-solo, GNN-Det+LSTM, GNN-VAE+LSTM):

```bash
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001
```

## Entrenamientos Selectivos

### Solo LSTM
```bash
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --skip-gnn-det \
    --skip-gvae \
    --epochs 50
```

### Solo GNN-Det+LSTM
```bash
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --skip-lstm \
    --skip-gvae \
    --epochs 50
```

### Solo GNN-VAE+LSTM
```bash
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --skip-lstm \
    --skip-gnn-det \
    --epochs 50
```

### GNN-Det + GNN-VAE (sin LSTM-solo)
```bash
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --skip-lstm \
    --epochs 50
```

## Visualización de Resultados

Genera gráficas de Loss y AUROC para los modelos entrenados:

```bash
python src/visualize_baseline.py
```

Esto crea:
- `visualizations/baseline_losses_latest.png` - Loss de train y validación
- `visualizations/baseline_auroc_latest.png` - AUROC de train y validación

## Parámetros Importantes

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--epochs` | 50 | Épocas de entrenamiento |
| `--batch-size` | 32 | Tamaño del batch |
| `--lr` | 0.001 | Learning rate |
| `--gnn-hidden` | 256 | Dimensión oculta del GNN |
| `--latent-dim` | 128 | Dimensión latente del VAE |
| `--lstm-hidden` | 256 | Dimensión oculta del LSTM |
| `--num-lstm-layers` | 2 | Número de capas LSTM |
| `--dropout` | 0.3 | Dropout rate |
| `--kl-weight` | 0.01 | Peso de pérdida KL (VAE) |
| `--score-threshold` | 0.5 | Threshold para etiquetas binarias |

## Archivos de Salida

Después de entrenar, se generan:

### Por Modelo (inmediatamente después del entrenamiento)
- `ablation_results/partial_lstm_solo_<timestamp>.json`
- `ablation_results/partial_gnn_det_lstm_<timestamp>.json`
- `ablation_results/partial_gnn_vae_lstm_<timestamp>.json`

Estructura:
```json
{
  "model": "LSTM-solo",
  "metrics": {
    "best_val_auroc": 0.85,
    "best_val_acc": 0.82,
    "best_val_f1": 0.78,
    "test_auroc": 0.84,
    "test_acc": 0.81,
    "test_f1": 0.77,
    "best_threshold": 0.5,
    "history": {
      "train_loss": [...],
      "val_loss": [...],
      "train_auroc": [...],
      "val_auroc": [...]
    }
  },
  "config": {...},
  "timestamp": "20250120_143022"
}
```

### Consolidado (al final)
- `ablation_results/ablation_results_<timestamp>.json`

## Checkpoints de Modelos

Los mejores modelos se guardan automáticamente:
- `best_lstm_baseline.pt`
- `best_gnn_det_lstm.pt`
- `best_gvae_lstm.pt`

## Notas de Memoria

Si tienes problemas de memoria:

1. **Reduce batch-size**: `--batch-size 16` o `--batch-size 8`
2. **Divide los archivos**: Usa `divide_and_conquer.py` para crear archivos más pequeños
3. **Entrena modelos por separado**: Usa los flags `--skip-*` para entrenar uno a la vez

## Ejemplo Completo de Workflow

```bash
# 1. Dividir archivos grandes (opcional, si tienes problemas de memoria)
python src/divide_and_conquer.py \
    --input-dir preprocessed_data/lstm_solo \
    --output-dir preprocessed_data/lstm_solo_divided

python src/divide_and_conquer.py \
    --input-dir preprocessed_data/gnn \
    --output-dir preprocessed_data/gnn_divided

# 2. Entrenar solo LSTM primero
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --skip-gnn-det \
    --skip-gvae \
    --epochs 50

# 3. Entrenar GNN-Det
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --skip-lstm \
    --skip-gvae \
    --epochs 50

# 4. Entrenar GNN-VAE
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --skip-lstm \
    --skip-gnn-det \
    --epochs 50

# 5. Visualizar resultados
python src/visualize_baseline.py
```
