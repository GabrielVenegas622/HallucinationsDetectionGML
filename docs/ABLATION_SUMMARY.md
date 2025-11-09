# Resumen Ejecutivo: Experimentos de Ablación

## 📊 Objetivo

Implementar una **estrategia de ablación científica** para probar la hipótesis central:

> La dinámica estructural secuencial a través de las capas es la señal clave para detectar alucinaciones.

## 🎯 Arquitectura de la Ablación

### Tabla Comparativa

| Modelo | Estructura de Grafo | Incertidumbre | Componentes |
|--------|---------------------|---------------|-------------|
| **LSTM-solo** | ❌ No | ❌ No | Solo secuencia temporal |
| **GNN-det+LSTM** | ✅ Sí (determinista) | ❌ No | GCN + LSTM |
| **GVAE+LSTM** | ✅ Sí (variacional) | ✅ Sí | GVAE + LSTM |

### Hipótesis a Probar

```
Si: GVAE+LSTM > GNN-det+LSTM > LSTM-solo
Entonces:
  1. Estructura del grafo aporta valor (GNN-det > LSTM)
  2. Modelado de incertidumbre aporta valor adicional (GVAE > GNN-det)
```

## 🔧 Implementación

### Archivo Principal
`src/baseline.py` (~1000 líneas)

### Componentes Clave

1. **Modelos** (3 clases):
   - `LSTMBaseline`: Baseline sin estructura
   - `GNNDetLSTM`: Con estructura determinista (CHARM-style)
   - `GVAELSTM`: Con estructura + variacional (propuesto)

2. **Dataset**:
   - `SequentialTraceDataset`: Organiza grafos por trace completo
   - `collate_sequential_batch`: Agrupa grafos por capa

3. **Entrenamiento**:
   - `train_lstm_baseline()`: Entrena LSTM-solo
   - `train_gnn_det_lstm()`: Entrena GNN-det+LSTM
   - `train_gvae_lstm()`: Entrena GVAE+LSTM con pérdida VAE

4. **Experimento Principal**:
   - `run_ablation_experiments()`: Ejecuta los 3 experimentos y compara

## 📈 Flujo de Datos

```
Traces (.pkl) → SequentialTraceDataset → Train/Val/Test Split
                                                ↓
                                         DataLoaders
                                                ↓
                        ┌───────────────────────┼───────────────────────┐
                        ↓                       ↓                       ↓
                  LSTM-solo              GNN-det+LSTM             GVAE+LSTM
                        ↓                       ↓                       ↓
              Promediar nodos           GCN por capa            GVAE por capa
                        ↓                       ↓                       ↓
              LSTM secuencial           LSTM secuencial         LSTM secuencial
                        ↓                       ↓                       ↓
                  Clasificación            Clasificación           Clasificación
                        ↓                       ↓                       ↓
                    Score                    Score              Score + VAE Loss
```

## 🎮 Uso Rápido

```bash
# Comando mínimo
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file "ground_truth_scores.csv"

# Comando completo con ajustes
python src/baseline.py \
    --data-pattern "traces_data/*triviaqa*.pkl" \
    --scores-file "ground_truth_triviaqa.csv" \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001 \
    --gnn-hidden 128 \
    --latent-dim 64 \
    --lstm-hidden 256 \
    --output-dir ./ablation_results
```

## 📊 Salidas Esperadas

### 1. Tabla de Resultados (stdout)
```
RESULTADOS FINALES - TABLA DE ABLACIÓN
================================================================
Modelo                    Best Val Loss       Best Val MAE
----------------------------------------------------------------
GVAE+LSTM                      0.1654             0.2543
GNN-det+LSTM                   0.1982             0.2987
LSTM-solo                      0.2341             0.3421
----------------------------------------------------------------
```

### 2. Verificación de Hipótesis (stdout)
```
VERIFICACIÓN DE HIPÓTESIS
================================================================
LSTM-solo:     0.2341
GNN-det+LSTM:  0.1982 (✓ mejor que LSTM-solo)
GVAE+LSTM:     0.1654 (✓ mejor que GNN-det+LSTM)

🎉 HIPÓTESIS CONFIRMADA:
   GVAE+LSTM > GNN-det+LSTM > LSTM-solo
   La estructura del grafo Y la incertidumbre variacional aportan valor.
```

### 3. Archivos Generados
```
ablation_results/
├── ablation_results_20250109_143022.json  # Métricas completas
├── best_lstm_baseline.pt                   # Modelo LSTM entrenado
├── best_gnn_det_lstm.pt                   # Modelo GNN-det entrenado
└── best_gvae_lstm.pt                      # Modelo GVAE entrenado
```

## 🔬 Detalles Técnicos

### Pérdidas

**LSTM-solo y GNN-det+LSTM:**
```python
Loss = MSE(predictions, scores)
```

**GVAE+LSTM:**
```python
Task_Loss = MSE(predictions, scores)
Recon_Loss = MSE(reconstructed, original)
KL_Loss = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
Total_Loss = Task_Loss + 0.1 * (Recon_Loss + kl_weight * KL_Loss)
```

### Arquitecturas Detalladas

**LSTM-solo:**
```
Input: [batch, num_layers, hidden_dim]
    ↓
BiLSTM(256 hidden, 2 layers)
    ↓
FC(512 → 128 → 64 → 1)
    ↓
Output: [batch, 1]
```

**GNN-det+LSTM:**
```
For each layer:
    Graph → GCN(hidden→128) → GCN(128→128) → GlobalMeanPool
    
Sequence of layer representations:
    ↓
BiLSTM(128 → 256 hidden, 2 layers)
    ↓
FC(512 → 128 → 64 → 1)
    ↓
Output: [batch, 1]
```

**GVAE+LSTM:**
```
For each layer:
    Graph → GCN → GlobalMeanPool → [μ, log(σ²)]
    z = μ + σ * ε  (reparameterization)
    reconstruction = Decoder(z)
    
Sequence of z:
    ↓
BiLSTM(64 → 256 hidden, 2 layers)
    ↓
FC(512 → 128 → 64 → 1)
    ↓
Output: [batch, 1] + VAE losses
```

## 💡 Ventajas de Este Enfoque

### 1. **Ablación Limpia**
Cada modelo añade **exactamente un componente**:
- LSTM-solo → GNN-det: Añade estructura
- GNN-det → GVAE: Añade incertidumbre

### 2. **Comparación Justa**
- Mismo dataset, mismo split
- Mismo número de capas LSTM
- Misma función de pérdida base (MSE)
- Mismo proceso de entrenamiento

### 3. **Interpretabilidad**
Si GVAE > GNN-det > LSTM:
- ✅ Sabemos que estructura aporta
- ✅ Sabemos que incertidumbre aporta
- ✅ Podemos cuantificar cada contribución

### 4. **Publicable**
Este es el formato estándar de ablación en papers de ML:
- Baseline simple
- Añadir componente A
- Añadir componente B
- Probar A > baseline y A+B > A

## ⚙️ Configuraciones Recomendadas

### Dataset Pequeño (<1000 traces)
```bash
--epochs 100 \
--batch-size 8 \
--lr 0.0005 \
--dropout 0.5 \
--gnn-hidden 64 \
--lstm-hidden 128
```

### Dataset Mediano (1000-5000 traces)
```bash
--epochs 50 \
--batch-size 16 \
--lr 0.001 \
--dropout 0.3 \
--gnn-hidden 128 \
--lstm-hidden 256
```

### Dataset Grande (>5000 traces)
```bash
--epochs 30 \
--batch-size 32 \
--lr 0.001 \
--dropout 0.2 \
--gnn-hidden 256 \
--lstm-hidden 512
```

## 🚀 Próximos Pasos

### 1. Ejecutar Experimentos
```bash
python src/baseline.py --data-pattern "traces_data/*.pkl" --scores-file scores.csv
```

### 2. Analizar Resultados
- Verificar si hipótesis se confirma
- Analizar curvas de aprendizaje
- Identificar overfitting/underfitting

### 3. Ajustar Hiperparámetros
Si resultados no son buenos:
- Aumentar `--epochs`
- Ajustar `--lr`
- Modificar `--dropout`
- Cambiar `--kl-weight` para GVAE

### 4. Visualizar
Crear gráficas de:
- Train/Val loss por época
- Comparación de MAE
- Análisis de errores

### 5. Reportar
Usar tabla de ablación en paper/presentación

## 📚 Archivos Relacionados

- `src/baseline.py` - Script principal
- `src/dataloader.py` - Dataset de grafos
- `src/trace_extractor.py` - Extracción de traces
- `src/trace_to_gt.py` - Generación de ground truth
- `docs/BASELINE_ABLATION_GUIDE.md` - Guía completa

## ✅ Checklist de Validación

Antes de ejecutar, verificar:

- [ ] Traces extraídos en `traces_data/`
- [ ] Ground truth generado (`ground_truth_scores.csv`)
- [ ] GPU disponible (recomendado)
- [ ] Suficiente espacio en disco (~1-2 GB para modelos)
- [ ] PyTorch y PyTorch Geometric instalados
- [ ] Dataset tiene al menos 100 traces para split razonable

## 🎓 Interpretación para Paper

### Si GVAE > GNN-det > LSTM:

> "Nuestros experimentos de ablación (Tabla X) demuestran que la incorporación 
> de estructura de grafo mejora significativamente el baseline (GNN-det vs LSTM: 
> X% de mejora en MAE). Además, el modelado variacional de la incertidumbre 
> aporta una mejora adicional (GVAE vs GNN-det: Y% de mejora), confirmando 
> nuestra hipótesis de que tanto la estructura como la incertidumbre son señales 
> clave para la detección de alucinaciones."

### Si GNN-det > LSTM pero GVAE ≈ GNN-det:

> "Los resultados de ablación (Tabla X) muestran que la estructura de grafo es 
> beneficiosa (X% de mejora), aunque el componente variacional no aporta mejoras 
> significativas en este contexto. Esto sugiere que la información determinista 
> del grafo es suficiente para capturar las señales relevantes."

## 🔗 Conexión con Hipótesis

Este experimento es el **Acto 1** de tu argumento científico:

**Acto 1 (Ablación)**: Probar que estructura + incertidumbre funcionan
**Acto 2 (Comparación)**: Superar a SOTA con toda la potencia
**Acto 3 (Análisis)**: Entender QUÉ capturan los modelos

Con `baseline.py` completas el Acto 1 de forma limpia y científica.
