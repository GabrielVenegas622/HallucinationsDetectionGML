# Análisis de Parámetros de los Modelos

## Problema Identificado

Los modelos GNN-det+LSTM y GVAE+LSTM tienen **significativamente menos parámetros** que el modelo LSTM-solo, lo cual representa una **desventaja injusta** en la comparación de ablación.

## Conteo de Parámetros Actual

Con la configuración por defecto (`lstm_hidden=128`, `gnn_hidden=128`, `hidden_dim=4096`):

- **LSTM-solo**: ~4.37M parámetros
- **GNN-det+LSTM**: ~0.88M parámetros  
- **GVAE+LSTM**: ~1.5M parámetros

**Ratio**: LSTM-solo tiene **5x más parámetros** que GNN-det+LSTM

## ¿Por qué ocurre esto?

### LSTM-solo
```
Input: [batch, num_layers, 4096] 
       ↓
LSTM(input_size=4096, hidden_size=128, bidirectional=True)
       ↓ (mayor parte de los parámetros aquí: 4.3M)
Clasificador MLP
       ↓
Output: [batch, 1]
```

**Parámetros del LSTM**: 4 × (4096 + 128 + 1) × 128 × 2 = **4.33M parámetros**

### GNN-det+LSTM
```
Input: [num_nodes, 4096] (por capa)
       ↓
GNN: 4096 → 128 (reducción dimensional)
       ↓ (540K parámetros)
Global pooling: [batch, 128] por capa
       ↓
LSTM(input_size=128, hidden_size=128, bidirectional=True)
       ↓ (solo 263K parámetros porque input=128)
Clasificador MLP
       ↓
Output: [batch, 1]
```

**Parámetros del LSTM**: 4 × (128 + 128 + 1) × 128 × 2 = **263K parámetros**

El GNN reduce primero la dimensión de 4096 → 128, entonces el LSTM subsecuente tiene **mucho menos parámetros** porque solo procesa vectores de 128 dimensiones.

## Soluciones Propuestas

### Opción 1: Aumentar `gnn_hidden` (RECOMENDADA)

Para igualar la capacidad de parámetros, aumenta `gnn_hidden` a **512** o **1024**:

```bash
python baseline.py \
  --preprocessed-dir preprocessed_data \
  --gnn-hidden 512 \
  --lstm-hidden 128 \
  --epochs 50
```

Esto dará a los modelos GNN ~3-4M parámetros, comparable con LSTM-solo.

### Opción 2: Reducir `lstm_hidden` para LSTM-solo

Modificar el código para que LSTM-solo también reduzca dimensión primero:

```python
# Agregar una capa de proyección antes del LSTM
self.projection = nn.Linear(hidden_dim, gnn_hidden)

# En forward:
layer_sequence = self.projection(layer_sequence)
lstm_out, (h_n, c_n) = self.lstm(layer_sequence)
```

### Opción 3: Reportar capacidad de parámetros

Si mantienes la arquitectura actual, **debes mencionar explícitamente** en el paper que:
- Los modelos están entrenados con diferente capacidad de parámetros
- El LSTM-solo tiene 5x más parámetros
- Esto puede influir en los resultados

## Recomendación Final

**Usa Opción 1**: Aumenta `gnn_hidden` a 512 o 1024 para una comparación justa. Esto es más fiel al espíritu del ablation study: comparar arquitecturas con capacidad similar.

Los hiperparámetros recomendados para comparación justa:

```bash
# Para ~3.5M parámetros en todos los modelos
python baseline.py \
  --preprocessed-dir preprocessed_data \
  --gnn-hidden 512 \
  --lstm-hidden 128 \
  --latent-dim 256 \
  --epochs 50 \
  --batch-size 16 \
  --lr 0.001
```

## Verificación

Para verificar el conteo de parámetros, el script imprime automáticamente:

```
Parámetros del modelo: 4,372,353
```

Asegúrate de que todos los modelos tengan un número similar (±20%) para una comparación justa.
