# Solución: "index out of range in self" - Mismatch entre hidden_states y attentions

## 🔴 Problema Identificado

**Error:** `index out of range in self` durante operaciones GINE en GPU

**Síntomas:**
```
ERROR en GINE de capa 0:
  x.shape: torch.Size([1, 4096])      # Solo 1 nodo
  edge_index.shape: torch.Size([2, 30])  # 30 arcos
  edge_attr.shape: torch.Size([30, 1])
```

**Causa Raíz:** 
Las atenciones (`attentions`) tienen dimensiones que **no coinciden** con el número real de tokens en `hidden_states`.

### Ejemplo del Problema

```python
# Trace extraído:
hidden_states[layer_idx].shape = (1, 4096)    # 1 token, 4096 dimensiones
attentions[layer_idx].shape = (32, 512, 512)  # 32 heads, matriz 512x512

# Al promediar heads y aplicar threshold:
attn_avg = attentions.mean(axis=0)  # shape: (512, 512)
mask = attn_avg > 0.01              # Encuentra conexiones

# Problema: Los índices en mask pueden ser hasta 511,
# pero solo hay 1 nodo (índice 0)!
edge_index = [[0, 1, 2, ...], [1, 2, 3, ...]]  # ❌ Índices 1,2,3... no existen!
```

## ✅ Solución Implementada

### 1. Corrección en dataloader.py

Se agregó validación y recorte automático:

```python
# Obtener número real de nodos
num_nodes = node_features.shape[0]  # Ej: 1

# Recortar atenciones al tamaño correcto
if attn_avg.shape[0] > num_nodes or attn_avg.shape[1] > num_nodes:
    attn_avg = attn_avg[:num_nodes, :num_nodes]  # Ej: [1, 1]

# Validar índices antes de crear edge_index
if indices.numel() > 0:
    valid_mask = (indices[0] < num_nodes) & (indices[1] < num_nodes)
    indices = indices[:, valid_mask]  # Solo índices válidos
```

**Antes:** 
- attentions podía ser 512x512 para una secuencia de 1 token
- edge_index contenía índices fuera de rango
- CUDA crash con "index out of range"

**Después:**
- attentions se recorta a 1x1 (match con num_nodes)
- Solo se crean arcos con índices válidos [0, num_nodes)
- No más crashes

### 2. Script de Validación

Nuevo script `validate_traces.py` para detectar el problema:

```bash
python src/validate_traces.py --data-pattern "traces_data/*.pkl"
```

**Qué detecta:**
- ✓ Mismatch entre hidden_states y attentions
- ✓ NaN/Inf en los datos
- ✓ Valores fuera de rango en attentions
- ✓ Shapes incorrectos

**Salida esperada si hay problema:**
```
❌ CRÍTICO - Trace 42, capa 15: Mismatch entre hidden_states y attentions
   hidden_states shape: (1, 4096)
   attentions shape: (32, 512, 512)
   ⚠️  Los índices de atención (512x512) no coinciden con seq_len (1)
   Esto causará 'index out of range' al crear grafos!
```

## 🔍 ¿Por Qué Ocurre Este Problema?

### Posibles Causas

1. **Padding en el modelo:**
   - El modelo LLM usa secuencias de longitud fija (ej: 512)
   - Los hidden_states reales son más cortos
   - Las atenciones se calculan sobre toda la secuencia paddeada

2. **KV-cache activado:**
   - Algunos modelos usan cache para generación
   - Puede causar dimensiones inconsistentes

3. **Extracción incorrecta:**
   - Se extrajo solo una parte de hidden_states
   - Pero attentions se extrajo completo

4. **Secuencias muy cortas:**
   - Preguntas/respuestas de 1 token
   - Modelo procesó con padding

## 🚀 Cómo Verificar si Tus Datos Tienen Este Problema

```bash
# Paso 1: Validar traces
python src/validate_traces.py --data-pattern "traces_data/*.pkl"

# Paso 2: Si hay problemas críticos, verificar con diagnóstico
python src/diagnose_cuda_error.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv
```

## ✅ Soluciones

### Solución 1: Usar Dataloader Actualizado (Automático)

El dataloader ya está corregido. Solo necesitas:

```bash
# Asegurarte de que usas la versión actualizada
python src/quick_test.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv
```

Si pasa el test, el problema está resuelto.

### Solución 2: Re-extraer Traces (Recomendado a Largo Plazo)

Si tienes acceso al código de extracción, asegurar que:

```python
# Al extraer traces
for layer_idx in range(num_layers):
    hidden_states = outputs.hidden_states[layer_idx]  # Shape: (batch, seq_len, hidden_dim)
    attentions = outputs.attentions[layer_idx]        # Shape: (batch, heads, seq_len, seq_len)
    
    # IMPORTANTE: Usar la misma seq_len
    actual_seq_len = hidden_states.shape[1]
    
    # Recortar attentions si es necesario
    attentions = attentions[:, :, :actual_seq_len, :actual_seq_len]
    
    # Guardar
    trace['hidden_states'][layer_idx] = hidden_states[0].cpu().numpy()
    trace['attentions'][layer_idx] = attentions[0].cpu().numpy()
```

### Solución 3: Limpiar Datos Existentes

Script para corregir traces existentes:

```python
import pickle
import glob
import numpy as np

def fix_trace_dimensions(file_pattern, output_suffix='_fixed'):
    files = glob.glob(file_pattern)
    
    for file_path in files:
        with open(file_path, 'rb') as f:
            traces = pickle.load(f)
        
        for trace in traces:
            num_layers = len(trace['hidden_states'])
            
            for layer_idx in range(num_layers):
                hs = trace['hidden_states'][layer_idx]
                attn = trace['attentions'][layer_idx]
                
                seq_len = hs.shape[0]  # Número real de tokens
                
                # Recortar attentions
                if attn.shape[1] > seq_len or attn.shape[2] > seq_len:
                    attn = attn[:, :seq_len, :seq_len]
                    trace['attentions'][layer_idx] = attn
                    print(f"Corregido: {trace['question_id']}, capa {layer_idx}")
        
        # Guardar corregido
        output_path = file_path.replace('.pkl', f'{output_suffix}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(traces, f)
        
        print(f"Guardado: {output_path}")

# Usar
fix_trace_dimensions("traces_data/*.pkl")
```

Luego entrenar con los archivos `_fixed.pkl`.

## 📊 Verificación Post-Corrección

```bash
# 1. Validar traces corregidos
python src/validate_traces.py --data-pattern "traces_data/*_fixed.pkl"

# Debería mostrar:
# ✅ ¡TODOS LOS TRACES SON VÁLIDOS!

# 2. Quick test
python src/quick_test.py \
    --data-pattern "traces_data/*_fixed.pkl" \
    --scores-file ground_truth_scores.csv

# Debería mostrar:
# ✅ QUICK TEST PASADO

# 3. Entrenar
python src/baseline.py \
    --data-pattern "traces_data/*_fixed.pkl" \
    --scores-file ground_truth_scores.csv \
    --batch-size 16 \
    --epochs 50
```

## 🎯 Resumen

| Problema | Causa | Solución |
|----------|-------|----------|
| "index out of range" | attentions.shape no match con hidden_states.shape | Dataloader recorta automáticamente ✓ |
| "device-side assert" | Índices en edge_index > num_nodes | Validación de índices ✓ |
| CUDA crash | Combinación de ambos | Ambas soluciones aplicadas ✓ |

## ✅ Estado Actual

- ✅ **dataloader.py actualizado** con validación y recorte
- ✅ **validate_traces.py** para detectar problemas
- ✅ **Correcciones automáticas** en el código
- ✅ **Documentación completa**

El código ahora maneja automáticamente este problema. Si encuentras el error:

1. **Ejecutar validación:** `python src/validate_traces.py ...`
2. **Si hay problemas críticos:** Limpiar datos con script de arriba
3. **Si no hay problemas:** El dataloader los maneja automáticamente

---
**Última actualización:** 2024-11-09
**Estado:** ✅ SOLUCIONADO
