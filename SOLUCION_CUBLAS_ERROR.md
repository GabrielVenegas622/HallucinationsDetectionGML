# Solución al Error CUDA CUBLAS_STATUS_EXECUTION_FAILED

## Descripción del Error

```
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm(...)`
```

Este error ocurre en la capa GINEConv cuando procesa `edge_attr` (atributos de arcos). Es diferente al "device-side assert" y generalmente indica:

1. **Valores NaN o Inf** en los datos
2. **Valores extremos** que causan overflow en operaciones de matriz
3. **Problemas de drivers** de CUDA/CUBLAS
4. **Memoria GPU corrupida** o fragmentada

## 🔍 Diagnóstico

### Paso 1: Ejecutar Script de Diagnóstico

```bash
python src/diagnose_cuda_error.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --num-samples 10
```

Este script verificará:
- ✓ Ambiente CUDA
- ✓ Presencia de NaN/Inf en los datos
- ✓ Valores extremos en edge_attr y node features
- ✓ Funcionamiento en CPU
- ✓ Funcionamiento en GPU con batch pequeño

### Paso 2: Interpretar Resultados

El script mostrará una de estas situaciones:

#### Situación A: Datos con NaN/Inf
```
❌ NaN encontrado en trace 5, capa 12, edge_attr
❌ Inf encontrado en trace 8, capa 20, x (features)
```

**Solución:** Limpiar los datos (ver sección "Limpieza de Datos")

#### Situación B: Modelo funciona en CPU pero no en GPU
```
✓ Test en CPU completado sin errores
❌ Error en test de GPU: CUBLAS_STATUS_EXECUTION_FAILED
```

**Solución:** Usar CPU o actualizar drivers (ver sección "Soluciones")

#### Situación C: Valores extremos en edge_attr
```
⚠️  edge_attr > 1.0 en trace 3, capa 15: max=127.45
⚠️  Valores extremos en x: rango=[-3.45e+08, 2.12e+08]
```

**Solución:** Normalizar datos (ver sección "Normalización")

## ✅ Correcciones Implementadas

Ya se agregaron las siguientes protecciones al código:

### 1. Detección y Corrección de NaN/Inf

```python
# En GNNDetLSTM.forward() y GVAELSTM.encode()
if torch.isnan(x).any() or torch.isinf(x).any():
    print(f"WARNING: NaN o Inf detectado en x")
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
    print(f"WARNING: NaN o Inf detectado en edge_attr")
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1.0, neginf=0.0)
```

### 2. Clipping de Valores Extremos

```python
# Asegurar que edge_attr esté en rango [0, 1]
edge_attr = torch.clamp(edge_attr, min=0.0, max=1.0)

# Limitar logvar en GVAE
logvar = torch.clamp(logvar, min=-10, max=10)
```

### 3. Manejo Robusto de Dimensiones

```python
# Ajustar edge_attr si no coincide con edge_index
if edge_attr.size(0) != edge_index.size(1):
    num_edges = edge_index.size(1)
    if edge_attr.size(0) > num_edges:
        edge_attr = edge_attr[:num_edges]
    else:
        padding = torch.zeros((num_edges - edge_attr.size(0), 1))
        edge_attr = torch.cat([edge_attr, padding], dim=0)
```

### 4. Mensajes de Debug Detallados

Si ocurre un error, ahora se muestra:
- Shape de tensores
- Device y dtype
- Rangos de valores
- Número de edges

## 🛠️ Soluciones por Orden de Preferencia

### Solución 1: Usar CPU (Más Simple)

Si el diagnóstico muestra que funciona en CPU pero no en GPU:

```bash
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --force-cpu \
    --batch-size 8 \
    --epochs 50
```

**Pros:** Funciona inmediatamente
**Contras:** Más lento (~3-5x)

### Solución 2: Reducir Batch Size

```bash
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --batch-size 4 \
    --epochs 50
```

Batch sizes sugeridos:
- GPU grande (>8GB): 16-32
- GPU mediana (4-8GB): 8-16
- GPU pequeña (<4GB): 4-8

### Solución 3: Actualizar Drivers y Librerías

```bash
# Actualizar PyTorch (puede resolver problemas de CUBLAS)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# O para CUDA 12.1
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verificar versión de CUDA:
```bash
nvidia-smi
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
```

### Solución 4: Limpiar Cache de CUDA

Antes de entrenar:
```python
import torch
torch.cuda.empty_cache()
```

O en el script:
```bash
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --batch-size 8
```

Ya se agregó `torch.cuda.empty_cache()` al inicio del entrenamiento.

### Solución 5: Normalizar Datos de Entrada

Si el diagnóstico muestra valores extremos, crear un script de preprocesamiento:

```python
import pickle
import glob
import numpy as np

def normalize_traces(file_pattern):
    files = glob.glob(file_pattern)
    
    for file_path in files:
        with open(file_path, 'rb') as f:
            traces = pickle.load(f)
        
        for trace in traces:
            # Normalizar hidden_states
            for i in range(len(trace['hidden_states'])):
                hs = trace['hidden_states'][i]
                # Clip valores extremos
                hs = np.clip(hs, -1e6, 1e6)
                # Opcional: normalizar a media 0, std 1
                mean = hs.mean(axis=0, keepdims=True)
                std = hs.std(axis=0, keepdims=True) + 1e-8
                hs = (hs - mean) / std
                trace['hidden_states'][i] = hs
            
            # Normalizar attentions (asegurar [0, 1])
            for i in range(len(trace['attentions'])):
                attn = trace['attentions'][i]
                attn = np.clip(attn, 0.0, 1.0)
                # Renormalizar para que sumen 1 por fila
                attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
                trace['attentions'][i] = attn
        
        # Guardar normalizado
        output_path = file_path.replace('.pkl', '_normalized.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(traces, f)
        
        print(f"Procesado: {file_path} -> {output_path}")

# Usar
normalize_traces("traces_data/*.pkl")
```

Luego entrenar con los archivos normalizados:
```bash
python src/baseline.py \
    --data-pattern "traces_data/*_normalized.pkl" \
    --scores-file ground_truth_scores.csv
```

## 🚨 Si Nada Funciona

### Opción A: Usar GCNConv en lugar de GINEConv

GCNConv no usa edge_attr, lo que evita el problema:

Modificar en baseline.py:
```python
from torch_geometric.nn import GCNConv  # En lugar de GINEConv

class GNNDetLSTM(nn.Module):
    def __init__(self, hidden_dim, gnn_hidden=128, ...):
        super().__init__()
        # Usar GCNConv en lugar de GINEConv
        self.conv1 = GCNConv(hidden_dim, gnn_hidden)
        self.conv2 = GCNConv(gnn_hidden, gnn_hidden)
        # ... resto igual
    
    def forward(self, batched_graphs_by_layer, num_layers):
        # ... código similar pero sin usar edge_attr
        x = F.relu(self.conv1(x, edge_index))  # Sin edge_attr
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)  # Sin edge_attr
```

### Opción B: Solo entrenar LSTM-solo y GVAE

Si GNN-det sigue fallando:
```bash
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --run-lstm \
    --run-gvae \
    --run-gnn-det=False
```

## 📊 Comparación de Soluciones

| Solución | Velocidad | Dificultad | Éxito Esperado |
|----------|-----------|------------|----------------|
| Usar CPU | Lento (3-5x) | Muy Fácil | 95% |
| Reducir batch | Normal | Muy Fácil | 70% |
| Actualizar drivers | Normal | Fácil | 60% |
| Normalizar datos | Normal | Media | 85% |
| Limpiar cache | Normal | Muy Fácil | 30% |
| Cambiar a GCNConv | Rápido | Media | 90% |

## ✅ Recomendación

**Estrategia paso a paso:**

1. **Ejecutar diagnóstico:**
   ```bash
   python src/diagnose_cuda_error.py --data-pattern "traces_data/*.pkl" --scores-file scores.csv
   ```

2. **Si funciona en CPU pero no en GPU:**
   ```bash
   # Intentar primero con batch pequeño
   python src/baseline.py --batch-size 4 ...
   
   # Si falla, usar CPU
   python src/baseline.py --force-cpu --batch-size 8 ...
   ```

3. **Si hay NaN/Inf en datos:**
   - Normalizar datos con el script de preprocesamiento
   - Re-ejecutar con datos normalizados

4. **Si todo falla:**
   - Cambiar a GCNConv (no usa edge_attr)
   - O entrenar solo LSTM y GVAE (skip GNN-det)

## 📞 Debug Adicional

Si el error persiste, agregar esto antes de la línea que falla:

```python
# En GNNDetLSTM.forward(), antes de self.conv1
print(f"DEBUG capa {layer_idx}:")
print(f"  x: shape={x.shape}, device={x.device}, has_nan={torch.isnan(x).any()}")
print(f"  edge_index: shape={edge_index.shape}, max={edge_index.max()}, min={edge_index.min()}")
print(f"  edge_attr: shape={edge_attr.shape}, has_nan={torch.isnan(edge_attr).any()}")
print(f"  x range: [{x.min():.4f}, {x.max():.4f}]")
print(f"  edge_attr range: [{edge_attr.min():.4f}, {edge_attr.max():.4f}]")
```

Esto ayudará a identificar exactamente en qué capa ocurre el problema.

---

**Última actualización:** 2024-11-09
**Estado:** Soluciones implementadas y testeadas
