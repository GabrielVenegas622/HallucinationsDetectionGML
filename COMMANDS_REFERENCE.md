# Referencia de Comandos - Pipeline de Datos

## 🔄 Flujo de Trabajo Completo

### 1. Pre-procesamiento Inicial
Convierte archivos .pkl/.pkl.gz originales a formato optimizado:

```bash
python src/preprocess_for_training.py \
    --data-pattern "traces_data/*.pkl*" \
    --scores-file traces_data/gt_*.csv \
    --output-dir preprocessed_data \
    --attn-threshold 0.0 \
    --score-threshold 0.5
```

**Resultado:**
- `preprocessed_data/lstm_solo/` - Datos para LSTM-solo
- `preprocessed_data/gnn/` - Datos para GNN-det+LSTM y GVAE+LSTM

---

### 2. División de Archivos (Opcional, para memoria limitada)
Divide archivos grandes en chunks pequeños:

```bash
# Para LSTM
python src/divide_and_conquer.py \
    --input-dir preprocessed_data/lstm_solo \
    --output-dir preprocessed_data/lstm_solo_split \
    --traces-per-part 50

# Para GNN
python src/divide_and_conquer.py \
    --input-dir preprocessed_data/gnn \
    --output-dir preprocessed_data/gnn_split \
    --traces-per-part 50
```

**Notas:**
- `--traces-per-part 50` divide archivos de 250 traces en 5 partes de 50
- Reduce uso de memoria durante entrenamiento
- Permite usar más `num_workers` en DataLoader

---

### 3. Verificación del Split
Verifica que la división train/val/test sea correcta:

```bash
python src/test_data_split.py \
    --lstm-dir preprocessed_data/lstm_solo_split \
    --gnn-dir preprocessed_data/gnn_split
```

**Salida esperada:**
- Train: ~70% de archivos
- Val: ~15% de archivos
- Test: ~15% de archivos

---

### 4. Entrenamiento

#### 4.1 Entrenar todos los modelos
```bash
python src/baseline.py \
    --lstm-dir preprocessed_data/lstm_solo_split \
    --gnn-dir preprocessed_data/gnn_split \
    --batch-size 64 \
    --num-workers 4 \
    --epochs 50
```

#### 4.2 Entrenar desde GNN-det en adelante (skip LSTM-solo)
```bash
python src/baseline.py \
    --lstm-dir preprocessed_data/lstm_solo_split \
    --gnn-dir preprocessed_data/gnn_split \
    --skip-lstm \
    --batch-size 64 \
    --num-workers 4
```

#### 4.3 Entrenar solo un modelo específico
```bash
# Solo LSTM
python src/baseline.py --lstm-dir ... --models lstm

# Solo GNN-det+LSTM
python src/baseline.py --gnn-dir ... --models gnn-det

# Solo GVAE+LSTM
python src/baseline.py --gnn-dir ... --models gvae
```

---

### 5. Visualización de Resultados

```bash
python src/visualize_baseline.py
```

**Genera:**
- Gráficas de Loss (train y validation)
- Gráficas de AUROC (train y test)
- Comparación entre modelos (LSTM, GNN-det, GVAE)
- Archivos guardados en `visualizations/`

---

## 📊 Estructura de Archivos Resultantes

```
preprocessed_data/
├── lstm_solo/                          # Original (puede ser grande)
│   └── preprocessed_*.pt
├── lstm_solo_split/                    # Dividido (recomendado)
│   ├── preprocessed_*_part0.pt
│   ├── preprocessed_*_part1.pt
│   └── ...
├── gnn/                                # Original (puede ser grande)
│   └── preprocessed_*.pt
└── gnn_split/                          # Dividido (recomendado)
    ├── preprocessed_*_part0.pt
    └── ...

ablation_results/
├── lstm_solo_TIMESTAMP.json            # Resultados LSTM
├── gnn_det_lstm_TIMESTAMP.json         # Resultados GNN-det
├── gvae_lstm_TIMESTAMP.json            # Resultados GVAE
└── ablation_summary.json               # Resumen de todos

visualizations/
├── baseline_losses.png
├── baseline_auroc.png
└── ...
```

---

## 🔍 Troubleshooting

### Problema: "Out of Memory"
**Solución:** Usa archivos divididos y reduce batch_size
```bash
python src/divide_and_conquer.py --traces-per-part 25  # Más pequeño
python src/baseline.py --batch-size 32  # Reducir batch
```

### Problema: "Train set vacío" (0 archivos)
**Causa:** Solo hay 1 archivo en la carpeta
**Solución:** Divide primero con `divide_and_conquer.py`

### Problema: Entrenamiento muy lento
**Causa:** num_workers=0 o archivos muy grandes
**Solución:**
```bash
python src/divide_and_conquer.py ...  # Dividir primero
python src/baseline.py --num-workers 4  # Usar paralelismo
```

### Problema: Test set tiene 183 archivos (desbalanceado)
**Causa:** Bug en versión anterior (ya corregido)
**Solución:** Actualizar baseline.py (ya incluido en este fix)

---

## 📝 Notas Importantes

1. **Orden recomendado**: preprocess → divide → verify → train → visualize
2. **Seed fijo**: Se usa `random.seed(42)` para reproducibilidad
3. **Split a nivel de archivos**: No de traces individuales
4. **Guardado automático**: Cada modelo se guarda al terminar su entrenamiento
5. **Threshold óptimo**: Se calcula en validación usando Youden's J statistic
