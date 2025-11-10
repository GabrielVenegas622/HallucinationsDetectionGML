# 🎯 PASOS A SEGUIR AHORA (ACTUALIZADO)

## Situación Actual - PROBLEMA IDENTIFICADO ✓

Has encontrado el error **"index out of range in self"** causado por un **mismatch entre hidden_states y attentions**.

**Diagnóstico:**
```
x.shape: torch.Size([1, 4096])        # Solo 1 token
edge_index.shape: torch.Size([2, 30])  # Pero hay 30 arcos
```

**Causa:** Las atenciones tienen dimensiones mayores que el número real de tokens.

## ✅ SOLUCIÓN APLICADA

Ya se corrigió el `dataloader.py` para:
- ✓ Recortar attentions al tamaño de hidden_states
- ✓ Validar índices antes de crear edge_index
- ✓ Filtrar índices fuera de rango

## 🚀 Acción Inmediata (2 minutos)

### PASO 1: Validar Tus Datos

```bash
python src/validate_traces.py --data-pattern "traces_data/*.pkl"
```

**Esto te dirá:**
- Si hay mismatches críticos en tus datos
- Cuántos traces tienen el problema
- Qué hacer al respecto

### PASO 2A: Si la Validación Dice "TODOS VÁLIDOS"

```bash
# Ejecutar quick test
python src/quick_test.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv

# Si pasa, entrenar:
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --batch-size 16 \
    --epochs 50
```

### PASO 2B: Si Hay Problemas Críticos

El dataloader YA los maneja automáticamente, pero si quieres limpiar los datos permanentemente:

```python
# Crear archivo clean_traces.py
import pickle
import glob
import numpy as np

def fix_traces(pattern):
    files = glob.glob(pattern)
    for file_path in files:
        with open(file_path, 'rb') as f:
            traces = pickle.load(f)
        
        for trace in traces:
            for layer_idx in range(len(trace['hidden_states'])):
                hs = trace['hidden_states'][layer_idx]
                attn = trace['attentions'][layer_idx]
                
                seq_len = hs.shape[0]
                if attn.shape[1] > seq_len or attn.shape[2] > seq_len:
                    trace['attentions'][layer_idx] = attn[:, :seq_len, :seq_len]
        
        output = file_path.replace('.pkl', '_fixed.pkl')
        with open(output, 'wb') as f:
            pickle.dump(traces, f)
        print(f"Fixed: {output}")

fix_traces("traces_data/*.pkl")
```

Luego entrenar con `*_fixed.pkl`.

#### Opción A: Funciona en CPU pero no en GPU
**Solución: Usar CPU**
```bash
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --force-cpu \
    --batch-size 8 \
    --epochs 50 \
    --run-lstm \
    --run-gnn-det \
    --run-gvae
```

Esto será más lento (~3-5x) pero funcionará sin problemas.

#### Opción B: Hay NaN/Inf en los datos
**Solución: Normalizar datos**

1. Ver el script de normalización en `SOLUCION_CUBLAS_ERROR.md` (Sección "Solución 5")
2. Ejecutarlo sobre tus datos
3. Entrenar con datos normalizados

#### Opción C: Valores extremos pero no NaN
**Solución: El código ya los maneja**

Debería funcionar. Si no, reducir batch:
```bash
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --batch-size 4 \
    --epochs 50
```

## 🚀 Solución Rápida sin Diagnóstico

Si solo quieres que funcione YA:

```bash
# OPCIÓN 1: Entrenar solo LSTM (funciona seguro)
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --run-lstm \
    --run-gnn-det=False \
    --run-gvae=False \
    --batch-size 16 \
    --epochs 50

# OPCIÓN 2: Todo en CPU (más lento pero funcional)
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --force-cpu \
    --batch-size 8 \
    --epochs 50
```

## 🔧 Soluciones Alternativas

### Alternativa 1: Actualizar PyTorch

A veces el error se debe a versiones incompatibles:

```bash
# Para CUDA 11.8
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verificar versión instalada
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

### Alternativa 2: Reducir Complejidad del Modelo

```bash
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --gnn-hidden 64 \      # Reducido de 128
    --lstm-hidden 128 \    # Reducido de 256
    --batch-size 8 \
    --epochs 50
```

### Alternativa 3: Cambiar Arquitectura GNN

Editar `baseline.py` para usar GCNConv en lugar de GINEConv:

```python
# En la clase GNNDetLSTM, cambiar:
from torch_geometric.nn import GCNConv  # En lugar de GINEConv

self.conv1 = GCNConv(hidden_dim, gnn_hidden)
self.conv2 = GCNConv(gnn_hidden, gnn_hidden)

# En forward(), NO pasar edge_attr:
x = F.relu(self.conv1(x, edge_index))  # Sin edge_attr
x = F.dropout(x, p=0.2, training=self.training)
x = self.conv2(x, edge_index)  # Sin edge_attr
```

GCNConv no usa edge_attr, lo que evita el error CUBLAS en esa parte.

## 📊 Comparación de Soluciones

| Solución | Tiempo | Dificultad | Probabilidad de Éxito |
|----------|--------|------------|----------------------|
| Usar CPU | Inmediato | ⭐ Muy Fácil | 99% |
| Reducir batch | Inmediato | ⭐ Muy Fácil | 70% |
| Actualizar PyTorch | 5-10 min | ⭐⭐ Fácil | 60% |
| Normalizar datos | 10-20 min | ⭐⭐⭐ Media | 90% |
| Cambiar a GCNConv | 15-30 min | ⭐⭐⭐ Media | 95% |
| Solo LSTM | Inmediato | ⭐ Muy Fácil | 100% |

## ⏱️ Mi Recomendación

**Para obtener resultados HOY:**

1. **Ejecutar diagnóstico** (2 min):
   ```bash
   python src/diagnose_cuda_error.py \
       --data-pattern "traces_data/*.pkl" \
       --scores-file ground_truth_scores.csv
   ```

2. **Si dice "funciona en CPU"**, usar CPU (inmediato):
   ```bash
   python src/baseline.py \
       --data-pattern "traces_data/*.pkl" \
       --scores-file ground_truth_scores.csv \
       --force-cpu \
       --batch-size 8 \
       --epochs 50
   ```

3. **Mientras entrena en CPU**, preparar normalización de datos para futuro entrenamiento en GPU

**Para mejor rendimiento a LARGO PLAZO:**

1. Normalizar los datos (script en `SOLUCION_CUBLAS_ERROR.md`)
2. Entrenar con datos normalizados en GPU
3. O cambiar a GCNConv si la normalización no funciona

## 🎯 Comando Final Recomendado

```bash
# EJECUTAR ESTO PRIMERO (diagnóstico)
python src/diagnose_cuda_error.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv

# LUEGO, BASADO EN EL RESULTADO:

# Si funciona en CPU pero no GPU → Usar esto:
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --force-cpu \
    --batch-size 8 \
    --epochs 50 \
    --score-threshold 0.5

# Si hay problemas en los datos → Normalizar primero, luego entrenar

# Si funciona en ambos → Usar GPU:
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --batch-size 16 \
    --epochs 50 \
    --score-threshold 0.5
```

## 📞 Si Nada Funciona

1. **Capturar el output completo del diagnóstico**
2. **Ejecutar con debug:**
   ```bash
   python src/baseline.py ... --run-gnn-det 2>&1 | tee error_log.txt
   ```
3. **Revisar** `SOLUCION_CUBLAS_ERROR.md` para soluciones avanzadas
4. **Contactar** con el log completo

## ✅ Próximos Pasos Después de Resolver

Una vez que el entrenamiento funcione:

1. ✓ Monitorear métricas AUROC
2. ✓ Experimentar con diferentes `--score-threshold`
3. ✓ Comparar resultados de los 3 modelos
4. ✓ Guardar los mejores modelos

---
**ACCIÓN INMEDIATA:** Ejecutar `python src/diagnose_cuda_error.py ...`
**BACKUP PLAN:** Usar `--force-cpu`
**TIEMPO ESTIMADO:** 5 minutos para tener entrenamiento corriendo
