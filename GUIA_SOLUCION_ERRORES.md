# 🔧 Guía de Solución de Errores - baseline.py

## 📋 Resumen

Este documento cubre la solución de dos errores principales encontrados al ejecutar `baseline.py`:

1. **Error "device-side assert"** - ✅ SOLUCIONADO
2. **Error "CUBLAS_STATUS_EXECUTION_FAILED"** - ✅ SOLUCIONES DISPONIBLES

## 🚀 Quick Start - ¿Qué hacer primero?

### Opción 1: Diagnóstico Completo (Recomendado)

```bash
# Paso 1: Diagnóstico de CUDA
python src/diagnose_cuda_error.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv

# Paso 2: Quick Test
python src/quick_test.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv
```

### Opción 2: Entrenamiento Directo con Protecciones

```bash
# Si tienes problemas con GPU, usar CPU:
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --force-cpu \
    --batch-size 8 \
    --epochs 50

# Si GPU funciona bien:
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --batch-size 16 \
    --epochs 50
```

## 🛠️ Scripts Disponibles

### 1. diagnose_cuda_error.py ⚡ (NUEVO - Para error CUBLAS)

**Propósito:** Diagnosticar problemas de CUDA/CUBLAS

```bash
python src/diagnose_cuda_error.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv
```

**Qué hace:**
- ✓ Verifica ambiente CUDA
- ✓ Detecta NaN/Inf en los datos
- ✓ Identifica valores extremos
- ✓ Prueba modelo en CPU
- ✓ Prueba modelo en GPU
- ✓ Provee recomendaciones específicas

**Salida:** Diagnóstico completo + recomendaciones personalizadas

### 2. quick_test.py ⚡ (Para verificación rápida)

**Propósito:** Verificación rápida de que todo funciona

```bash
python src/quick_test.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --num-samples 100
```

**Tiempo:** 1-2 minutos

### 3. test_baseline.py 🔬 (Para tests exhaustivos)

**Propósito:** Tests completos con mini-entrenamiento

```bash
python src/test_baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --test-training
```

**Tiempo:** 5-10 minutos

## ❌ Error 1: "device-side assert" 

### Estado: ✅ SOLUCIONADO

Este error ya está completamente resuelto en la versión actual del código.

**Causa:** Inconsistencias en `edge_attr` (dimensiones, None, vacío)

**Solución implementada:**
- Validación automática en `collate_sequential_batch()`
- Manejo robusto en `GNNDetLSTM.forward()`
- Corrección automática de inconsistencias

**Ver:** `SOLUCION_DEVICE_ASSERT.md` para detalles técnicos

## ❌ Error 2: "CUBLAS_STATUS_EXECUTION_FAILED"

### Estado: ⚠️ SOLUCIONES DISPONIBLES

```
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm(...)`
```

Este error ocurre en operaciones de matriz en GPU, típicamente por:
- NaN/Inf en los datos
- Valores extremos
- Problemas de drivers CUDA
- Fragmentación de memoria GPU

### Solución Rápida (Más Fácil)

**Opción A: Usar CPU**
```bash
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --force-cpu \
    --batch-size 8 \
    --epochs 50
```

**Pros:** Funciona inmediatamente
**Contras:** ~3-5x más lento

**Opción B: Reducir Batch Size**
```bash
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --batch-size 4 \
    --epochs 50
```

### Solución Completa (Recomendada)

1. **Ejecutar diagnóstico:**
   ```bash
   python src/diagnose_cuda_error.py \
       --data-pattern "traces_data/*.pkl" \
       --scores-file ground_truth_scores.csv
   ```

2. **Seguir recomendaciones del diagnóstico:**
   - Si funciona en CPU pero no GPU → Usar `--force-cpu`
   - Si hay NaN/Inf en datos → Normalizar datos (ver abajo)
   - Si hay valores extremos → Aplicar clipping

3. **Normalizar datos (si es necesario):**
   Ver script de normalización en `SOLUCION_CUBLAS_ERROR.md`

**Ver:** `SOLUCION_CUBLAS_ERROR.md` para soluciones detalladas

## 📊 Tabla de Decisión Rápida

| Síntoma | Solución | Comando |
|---------|----------|---------|
| LSTM funciona, GNN falla | Usar CPU o reducir batch | `--force-cpu --batch-size 8` |
| "device-side assert" | Ya resuelto | Usar versión actual |
| "CUBLAS error" | Diagnóstico primero | `python src/diagnose_cuda_error.py ...` |
| NaN/Inf en datos | Normalizar datos | Ver `SOLUCION_CUBLAS_ERROR.md` |
| Out of memory | Reducir batch | `--batch-size 4` |
| Todo funciona | Entrenar normal | `--batch-size 16 --epochs 50` |

## 🎯 Workflow Recomendado

```bash
# 1. DIAGNÓSTICO (5 min)
python src/diagnose_cuda_error.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv

# 2. QUICK TEST (2 min)
python src/quick_test.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv

# 3a. Si todo OK - ENTRENAR EN GPU
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --batch-size 16 \
    --epochs 50

# 3b. Si hay problemas - ENTRENAR EN CPU
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --force-cpu \
    --batch-size 8 \
    --epochs 50
```

## 🔍 Verificar que Correcciones Están Aplicadas

```bash
# Verificar manejo de edge_attr
grep -n "Manejo seguro de edge_attr" src/baseline.py

# Debería mostrar 2 líneas (GNNDetLSTM y GVAELSTM)
# Si no, actualizar baseline.py
```

## 📚 Documentación Disponible

1. **SOLUCION_CUBLAS_ERROR.md** - Soluciones para error CUBLAS (NUEVO)
2. **SOLUCION_DEVICE_ASSERT.md** - Soluciones para device-side assert
3. **TEST_README.md** - Guía de scripts de testing
4. **RESUMEN_CORRECCIONES.md** - Resumen ejecutivo de correcciones
5. **CAMBIOS_HALOSCOPE.md** - Cambios a metodología HaloScope
6. **CHANGELOG_FIXES.md** - Registro completo de cambios

## 🆕 Nuevas Características

### 1. Modo CPU Forzado
```bash
--force-cpu  # Usar CPU aunque GPU esté disponible
```

### 2. Validación de NaN/Inf
El código ahora detecta y corrige automáticamente:
- NaN → 0.0
- Inf → valores acotados (1e6)
- Muestra warnings cuando hace correcciones

### 3. Clipping de Valores
- edge_attr → [0.0, 1.0]
- logvar → [-10, 10]

### 4. Debug Mejorado
Mensajes detallados cuando ocurre un error:
```
ERROR en GINE de capa 15:
  x.shape: torch.Size([512, 4096])
  edge_index.shape: torch.Size([2, 2048])
  edge_attr.shape: torch.Size([2048, 1])
  Rango de x: [-12.3456, 15.7890]
```

## ⚙️ Parámetros Importantes

```bash
--force-cpu              # Forzar CPU (si GPU da problemas)
--batch-size 4           # Batch pequeño para GPUs con poca memoria
--batch-size 8           # Batch mediano (recomendado para CPU)
--batch-size 16          # Batch grande (recomendado para GPU)
--score-threshold 0.5    # Threshold para clasificación binaria
--attn-threshold 0.01    # Threshold para crear arcos de atención
--epochs 50              # Número de épocas
```

## 🐛 Troubleshooting

### Problema: "No module named 'baseline'"
```bash
# Asegurarse de estar en el directorio correcto
cd /path/to/HallucinationsDetectionGML
python src/quick_test.py ...
```

### Problema: "CUDA out of memory"
```bash
# Solución 1: Reducir batch
--batch-size 4

# Solución 2: Usar CPU
--force-cpu --batch-size 8
```

### Problema: Warnings de NaN/Inf durante entrenamiento
```
WARNING: NaN o Inf detectado en edge_attr de capa 12
```

**Solución:** Los datos tienen valores corruptos. Ejecutar:
```bash
python src/diagnose_cuda_error.py ...
```

Y seguir recomendaciones para normalizar datos.

### Problema: Entrenamiento muy lento
```bash
# Si estás en CPU, es normal (3-5x más lento)
# Para acelerar:
--batch-size 16  # Aumentar batch size en CPU
--num-lstm-layers 1  # Reducir complejidad del modelo
```

## 📞 Soporte

1. **Revisar documentación:** Archivos .md en el directorio raíz
2. **Ejecutar diagnóstico:** `python src/diagnose_cuda_error.py ...`
3. **Revisar logs:** Los warnings indican qué está pasando
4. **Contactar con traceback completo** si el problema persiste

## ✅ Checklist Pre-Entrenamiento

- [ ] Ejecutar `diagnose_cuda_error.py`
- [ ] Ejecutar `quick_test.py`
- [ ] Verificar que no hay warnings de NaN/Inf
- [ ] Confirmar batch-size apropiado para tu GPU/CPU
- [ ] Limpiar cache de CUDA si usas GPU
- [ ] Tener suficiente espacio en disco para resultados

## 🎯 Resultado Esperado

Si todo funciona correctamente:

```
============================================================
QUICK TEST PASADO - Todo funciona correctamente!
   Puedes proceder con el entrenamiento completo.
============================================================

Entrenamiento:
Epoch 1: Train Loss=0.6234, Val Loss=0.5892, AUROC=0.7234, Acc=0.6850, F1=0.6512
Epoch 2: Train Loss=0.5892, Val Loss=0.5645, AUROC=0.7456, Acc=0.7012, F1=0.6734
...
```

---
**Versión:** 2.1
**Última actualización:** 2024-11-09
**Estado:** Estable con múltiples soluciones disponibles
