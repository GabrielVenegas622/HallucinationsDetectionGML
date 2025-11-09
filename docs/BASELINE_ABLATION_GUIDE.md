# Guía de Experimentos de Ablación: `baseline.py`

## Descripción General

El archivo `baseline.py` implementa la **estrategia de ablación científica** para probar la hipótesis central del trabajo:

> **Hipótesis**: La dinámica estructural secuencial a través de las capas del modelo de lenguaje es la señal clave para detectar alucinaciones.

## Arquitecturas Implementadas

### 1. LSTM-solo (Baseline)
**Sin estructura de grafo**

```
Secuencia de capas → LSTM → Clasificación
```

- **Input**: Secuencia de representaciones por capa (promedio de hidden states)
- **Procesamiento**: LSTM bidireccional captura dependencias temporales entre capas
- **Output**: Score de hallucination
- **Ignora**: Estructura de atención entre tokens

**Propósito**: Establecer baseline sin usar información estructural del grafo.

### 2. GNN-det+LSTM (CHARM-style)
**Con estructura de grafo determinista**

```
Capa 1: GCN → repr₁
Capa 2: GCN → repr₂    } → Secuencia → LSTM → Clasificación
...
Capa N: GCN → reprₙ
```

- **Input**: Grafos por capa (nodos = tokens, arcos = atención)
- **Procesamiento**: 
  - GCN extrae representación considerando estructura
  - LSTM captura dinámica entre capas
- **Output**: Score de hallucination
- **Usa**: Estructura del grafo de forma determinista

**Propósito**: Probar si la estructura del grafo aporta valor.

### 3. GVAE+LSTM (Modelo Propuesto)
**Con estructura + modelado de incertidumbre variacional**

```
Capa 1: GVAE Encoder → z₁ ~ N(μ₁, σ₁²)
Capa 2: GVAE Encoder → z₂ ~ N(μ₂, σ₂²)    } → Secuencia latente → LSTM → Clasificación
...
Capa N: GVAE Encoder → zₙ ~ N(μₙ, σₙ²)
```

- **Input**: Grafos por capa
- **Procesamiento**:
  - GVAE Encoder proyecta a espacio latente con incertidumbre
  - Reparameterization trick: z = μ + σ·ε
  - LSTM sobre secuencia latente
  - Regularización con reconstrucción y KL divergence
- **Output**: Score + distribuciones latentes
- **Usa**: Estructura + incertidumbre probabilística

**Propósito**: Probar si modelar incertidumbre mejora la detección.

## Resultado Esperado

Si la hipótesis es correcta:

```
GVAE+LSTM > GNN-det+LSTM > LSTM-solo
```

Esto probaría que:
1. **Estructura del grafo** aporta valor (GNN-det > LSTM)
2. **Incertidumbre variacional** aporta valor adicional (GVAE > GNN-det)

## Uso

### Comando Básico

```bash
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file "ground_truth_scores.csv" \
    --epochs 50 \
    --batch-size 16
```

### Parámetros Principales

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--data-pattern` | Patrón glob para archivos .pkl | **Requerido** |
| `--scores-file` | CSV con scores BLEURT | **Requerido** |
| `--epochs` | Número de épocas de entrenamiento | 50 |
| `--batch-size` | Tamaño del batch | 16 |
| `--lr` | Learning rate | 0.001 |

### Parámetros de Arquitectura

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--gnn-hidden` | Dimensión oculta de GNN | 128 |
| `--latent-dim` | Dimensión latente para GVAE | 64 |
| `--lstm-hidden` | Dimensión oculta de LSTM | 256 |
| `--num-lstm-layers` | Número de capas LSTM | 2 |
| `--dropout` | Tasa de dropout | 0.3 |
| `--kl-weight` | Peso para KL divergence en GVAE | 0.001 |

### Control de Experimentos

```bash
# Ejecutar solo LSTM baseline
python src/baseline.py ... --run-lstm --no-run-gnn-det --no-run-gvae

# Ejecutar solo comparación GNN vs GVAE
python src/baseline.py ... --no-run-lstm --run-gnn-det --run-gvae

# Ejecutar todos (default)
python src/baseline.py ... --run-lstm --run-gnn-det --run-gvae
```

## Pipeline de Ejecución

### 1. Carga de Datos
```
Traces (.pkl) + Scores BLEURT → SequentialTraceDataset
                               ↓
                    Split: Train (70%) / Val (15%) / Test (15%)
                               ↓
                         DataLoaders con collate_fn
```

### 2. Entrenamiento de Cada Modelo

**Para cada modelo:**
- Inicialización de arquitectura
- Loop de entrenamiento (epochs):
  - Forward pass
  - Cálculo de pérdida
  - Backward pass y optimización
  - Validación
  - Guardar mejor modelo
- Retorno de historia de métricas

### 3. Comparación y Resultados

**Tabla de Ablación:**
```
Modelo                Best Val Loss    Best Val MAE
---------------------------------------------------------
LSTM-solo                  0.2341           0.3421
GNN-det+LSTM               0.1982           0.2987
GVAE+LSTM                  0.1654           0.2543
```

**Verificación de Hipótesis:**
```
✓ GNN-det+LSTM mejor que LSTM-solo → Estructura aporta valor
✓ GVAE+LSTM mejor que GNN-det+LSTM → Incertidumbre aporta valor
🎉 HIPÓTESIS CONFIRMADA
```

## Funciones de Pérdida

### LSTM-solo y GNN-det+LSTM
```python
Loss = MSE(predictions, scores)
```

### GVAE+LSTM
```python
Task_Loss = MSE(predictions, scores)
VAE_Loss = Reconstruction_Loss + KL_weight * KL_Divergence
Total_Loss = Task_Loss + 0.1 * VAE_Loss
```

Donde:
- **Reconstruction Loss**: MSE entre original y reconstrucción del grafo
- **KL Divergence**: -0.5 * Σ(1 + log(σ²) - μ² - σ²)

## Salidas

### Durante Entrenamiento

```
================================================================================
EXPERIMENTOS DE ABLACIÓN - PRUEBA DE HIPÓTESIS
================================================================================

Hipótesis: La dinámica estructural secuencial a través de las capas
           es la señal clave para detectar alucinaciones.

Modelos a comparar:
  1. LSTM-solo (Baseline sin estructura)
  2. GNN-det+LSTM (Con estructura determinista)
  3. GVAE+LSTM (Con estructura + incertidumbre variacional)

Esperado: GVAE+LSTM > GNN-det+LSTM > LSTM-solo
================================================================================

Dispositivo: cuda

Cargando dataset...
Dataset secuencial creado:
  - 5000 traces
  - 32 capas por trace
  - 5000 scores cargados

Split del dataset:
  Train: 3500
  Val: 750
  Test: 750

Dimensión de hidden states: 4096

================================================================================
EXPERIMENTO 1: LSTM-solo (Baseline)
================================================================================
Parámetros del modelo: 5,234,689

Epoch 1/50: Train Loss=0.3421, Val Loss=0.2987, Val MAE=0.4123
Epoch 2/50: Train Loss=0.2834, Val Loss=0.2654, Val MAE=0.3876
...
```

### Archivo de Resultados JSON

`ablation_results/ablation_results_20250109_143022.json`:

```json
{
  "LSTM-solo": {
    "best_val_loss": 0.2341,
    "best_val_mae": 0.3421,
    "history": {
      "train_loss": [0.3421, 0.2834, ...],
      "val_loss": [0.2987, 0.2654, ...],
      "val_mae": [0.4123, 0.3876, ...]
    }
  },
  "GNN-det+LSTM": {
    "best_val_loss": 0.1982,
    "best_val_mae": 0.2987,
    "history": { ... }
  },
  "GVAE+LSTM": {
    "best_val_loss": 0.1654,
    "best_val_mae": 0.2543,
    "history": { ... }
  }
}
```

### Modelos Guardados

- `best_lstm_baseline.pt` - Mejor modelo LSTM
- `best_gnn_det_lstm.pt` - Mejor modelo GNN-det
- `best_gvae_lstm.pt` - Mejor modelo GVAE

## Ejemplo Completo de Workflow

```bash
# 1. Extraer traces (si no lo has hecho)
python src/trace_extractor.py \
    --model llama2_chat_7B \
    --dataset triviaqa \
    --num-samples 5000

# 2. Generar ground truth scores
python src/trace_to_gt.py \
    --dataset triviaqa \
    --traces-dir ./traces_data \
    --output ground_truth_triviaqa.csv

# 3. Ejecutar experimentos de ablación
python src/baseline.py \
    --data-pattern "traces_data/*triviaqa*.pkl" \
    --scores-file ground_truth_triviaqa.csv \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001

# 4. Analizar resultados
ls ablation_results/
# ablation_results_20250109_143022.json
# best_lstm_baseline.pt
# best_gnn_det_lstm.pt
# best_gvae_lstm.pt
```

## Interpretación de Resultados

### Escenario 1: Hipótesis Confirmada ✓
```
GVAE+LSTM (0.165) < GNN-det+LSTM (0.198) < LSTM-solo (0.234)
```
**Conclusión**: Estructura Y incertidumbre aportan valor incremental.

### Escenario 2: Hipótesis Parcial ⚠️
```
GNN-det+LSTM (0.198) < LSTM-solo (0.234)
GVAE+LSTM (0.201) ≈ GNN-det+LSTM (0.198)
```
**Conclusión**: Estructura aporta valor, pero incertidumbre variacional no mejora significativamente. Revisar hiperparámetros de GVAE.

### Escenario 3: Hipótesis Rechazada ✗
```
LSTM-solo (0.234) < GNN-det+LSTM (0.265)
```
**Conclusión**: Revisar implementación o considerar que la estructura no aporta en este contexto específico.

## Consideraciones Técnicas

### Memoria GPU
- **LSTM-solo**: ~2-3 GB
- **GNN-det+LSTM**: ~4-6 GB
- **GVAE+LSTM**: ~5-8 GB

Para GPUs pequeñas, reducir `--batch-size`.

### Tiempo de Entrenamiento
Con 5000 traces, 32 capas, 50 épocas:
- **LSTM-solo**: ~30-45 min
- **GNN-det+LSTM**: ~1-2 horas
- **GVAE+LSTM**: ~2-3 horas

### Hiperparámetros Recomendados

**Para dataset pequeño (<1000 traces):**
```bash
--epochs 100 --lr 0.0005 --dropout 0.5
```

**Para dataset grande (>10000 traces):**
```bash
--epochs 30 --lr 0.001 --dropout 0.3 --batch-size 32
```

## Troubleshooting

### Error: CUDA out of memory
```bash
# Reducir batch size
--batch-size 8

# O reducir dimensiones
--gnn-hidden 64 --lstm-hidden 128
```

### Overfitting evidente
```bash
# Aumentar regularización
--dropout 0.5 --kl-weight 0.01

# O reducir capacidad
--num-lstm-layers 1
```

### Underfitting (loss no baja)
```bash
# Aumentar capacidad
--gnn-hidden 256 --lstm-hidden 512

# O entrenar más
--epochs 100 --lr 0.0001
```

## Extensiones Futuras

1. **Mecanismos de Atención**: Añadir attention entre capas en LSTM
2. **Graph Attention Networks**: Reemplazar GCN por GAT
3. **Ensemble**: Combinar predicciones de los 3 modelos
4. **Análisis de Incertidumbre**: Usar σ² del GVAE para calibración

## Referencias

- **VAE**: Kingma & Welling (2013) - Auto-Encoding Variational Bayes
- **GCN**: Kipf & Welling (2016) - Semi-Supervised Classification with GCNs
- **CHARM**: Manakul et al. (2023) - Hallucination detection baseline
