# Implementación de Procesamiento por Batches

## Resumen de Mejoras

Se ha actualizado el sistema de extracción de trazas para procesar y guardar datos en **batches** de 500 traces cada uno, optimizando el uso de memoria RAM.

---

## Motivación

**Problema Original:**
- Cargar 100+ traces en memoria consumía >10 GB de RAM
- No escalable para datasets grandes (TriviaQA completo = 87k ejemplos)
- Pérdida total de datos ante fallos durante procesamiento

**Solución Implementada:**
- Guardado incremental en batches de 500 traces
- Uso máximo de RAM: ~5-6 GB por batch
- Recuperación ante fallos: batches ya guardados se conservan

---

## Cambios Técnicos

### 1. `src/trace_extractor.py`

**Nuevas características:**
- Variable `BATCH_SIZE = 500` (configurable)
- Guardado automático cada 500 traces
- Nomenclatura de archivos: `trivia_qa_traces_batch_XXXX.pkl`
- Garbage collection explícito después de guardar cada batch
- IDs mejorados:
  - `example_id`: ID dentro del batch (0-499)
  - `global_example_id`: ID global en el dataset completo
  - `batch_number`: Número del batch correspondiente

**Ejemplo de salida:**
```
./traces_data/
├── trivia_qa_traces_batch_0000.pkl  # Traces 0-499
├── trivia_qa_traces_batch_0001.pkl  # Traces 500-999
├── trivia_qa_traces_batch_0002.pkl  # Traces 1000-1499
└── ...
```

### 2. `src/inspect_traces.py`

**Actualizaciones:**
- Soporte para múltiples archivos batch
- Estadísticas globales calculadas iterando sobre batches
- Comparación de archivos en formato antiguo vs. batch
- Análisis de ejemplos de diferentes batches

### 3. `src/batch_loader.py` (NUEVO)

**Clase principal: `TraceBatchLoader`**

Utilidad para trabajar eficientemente con múltiples batches sin cargar todo en memoria.

**Métodos principales:**

```python
loader = TraceBatchLoader("./traces_data")

# Obtener información sin cargar datos
info = loader.get_batch_info()

# Iterar sobre batches (carga uno a la vez)
for batch in loader.iter_batches():
    process(batch)

# Iterar sobre traces individuales
for trace in loader.iter_traces():
    print(trace['question'])

# Cargar batch específico
batch_0 = loader.get_batch(0)

# Buscar por ID global
trace = loader.get_trace_by_global_id(1234)
```

**Funciones auxiliares:**

- `merge_batches()`: Combina todos los batches en un archivo (⚠️ requiere RAM)
- `extract_batch_subset()`: Extrae solo algunos batches específicos

---

## Estimaciones de Recursos

### Por Trace Individual
- **Tamaño promedio**: ~10 MB (varía según longitud de respuesta)
- **RAM durante extracción**: ~130 MB (modelo + trace temporal)

### Por Batch (500 traces)
- **Tamaño en disco**: ~5 GB
- **RAM máxima**: ~6 GB (modelo 4GB + batch 2GB)

### Dataset Completo TriviaQA
- **Total ejemplos**: ~87,000
- **Batches necesarios**: 174 archivos
- **Espacio total en disco**: ~870 GB
- **RAM necesaria**: Solo 6 GB (procesa de a uno)

---

## Ventajas del Sistema de Batches

### 1. Escalabilidad
✅ Procesa datasets arbitrariamente grandes  
✅ RAM constante independiente del tamaño del dataset  
✅ Solo requiere espacio en disco

### 2. Robustez
✅ Recuperación ante fallos: batches ya guardados se conservan  
✅ Reinicio desde batch N (no desde cero)  
✅ Validación incremental

### 3. Flexibilidad
✅ Carga selectiva de datos (solo los batches necesarios)  
✅ Procesamiento paralelo posible (diferentes batches en diferentes GPUs)  
✅ Fácil división train/val/test

### 4. Eficiencia
✅ Garbage collection explícito libera memoria  
✅ Iteradores evitan cargas innecesarias  
✅ Operaciones lazy cuando es posible

---

## Uso Recomendado

### Para Pruebas (Primera Vez)
```python
# En trace_extractor.py, línea ~148
num_samples = 1000  # Solo 1000 ejemplos = 2 batches
```

Ejecutar:
```bash
python src/trace_extractor.py
```

Resultado esperado:
- 2 archivos batch (~10 GB total)
- Tiempo: ~30 minutos
- RAM: 6 GB máximo

### Para Dataset Completo
```python
# En trace_extractor.py, línea ~148
num_samples = None  # Procesar todo TriviaQA
```

Ejecutar (recomendado usar `nohup` o `screen`):
```bash
nohup python src/trace_extractor.py > extraction.log 2>&1 &
```

Resultado esperado:
- ~174 archivos batch (~870 GB total)
- Tiempo: ~2-3 días (depende de GPU)
- RAM: 6 GB máximo (constante)

### Para Inspeccionar Datos
```bash
# Ver estadísticas de todos los batches
python src/inspect_traces.py

# Usar programáticamente
python src/batch_loader.py
```

---

## Ejemplos de Código

### Ejemplo 1: Procesar todos los traces sin cargar todo en memoria

```python
from src.batch_loader import TraceBatchLoader

loader = TraceBatchLoader()

# Calcular estadística global
total_tokens = 0
count = 0

for trace in loader.iter_traces():
    num_tokens = len(trace['tokens']) - trace['prompt_length']
    total_tokens += num_tokens
    count += 1

print(f"Promedio de tokens: {total_tokens / count:.2f}")
```

### Ejemplo 2: Entrenar modelo batch por batch

```python
from src.batch_loader import TraceBatchLoader

loader = TraceBatchLoader()

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(loader.iter_batches()):
        # Procesar batch completo (500 traces)
        graphs = build_graphs_from_batch(batch)
        loss = model.train_step(graphs)
        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss}")
```

### Ejemplo 3: Crear splits train/val/test

```python
from src.batch_loader import TraceBatchLoader, extract_batch_subset

loader = TraceBatchLoader()
total_batches = len(loader)

# 80% train, 10% val, 10% test
train_batches = list(range(0, int(0.8 * total_batches)))
val_batches = list(range(int(0.8 * total_batches), int(0.9 * total_batches)))
test_batches = list(range(int(0.9 * total_batches), total_batches))

# Extraer splits
extract_batch_subset(train_batches, "./traces_data/train_set.pkl")
extract_batch_subset(val_batches, "./traces_data/val_set.pkl")
extract_batch_subset(test_batches, "./traces_data/test_set.pkl")
```

---

## Verificación de Implementación

Todos los archivos pasan verificación de sintaxis:
```bash
✅ src/trace_extractor.py
✅ src/inspect_traces.py
✅ src/batch_loader.py
```

---

## Archivos Actualizados

1. **`src/trace_extractor.py`** - Lógica de batching implementada
2. **`src/inspect_traces.py`** - Soporte para múltiples batches
3. **`src/batch_loader.py`** - Nueva utilidad (clase + funciones)
4. **`src/README_trace_extractor.md`** - Documentación actualizada

---

## Próximos Pasos Recomendados

1. **Probar con 1000 ejemplos** (2 batches) para validar
2. **Verificar tamaños de archivo** generados
3. **Usar `batch_loader.py`** para cargar y explorar datos
4. **Implementar dataloader para grafos** usando iteradores de batch
5. **Escalar a dataset completo** si recursos lo permiten

---

## Notas Importantes

⚠️ **Espacio en disco**: Asegúrate de tener suficiente espacio antes de procesar el dataset completo  
⚠️ **Tiempo de procesamiento**: El dataset completo puede tomar días  
⚠️ **Checkpointing**: Los batches guardados actúan como checkpoints automáticos  
⚠️ **RAM vs Disco**: Trade-off favorable: baja RAM, más disco

---

## Soporte y Troubleshooting

### "Out of Memory" durante extracción
- Reducir `BATCH_SIZE` (ej: 250 traces)
- Verificar que `gc.collect()` se ejecute
- Monitorear con `nvidia-smi` (GPU) o `htop` (CPU)

### "No space left on device"
- Verificar espacio: `df -h`
- Procesar menos ejemplos: ajustar `num_samples`
- Usar disco externo o almacenamiento en red

### Recuperación después de fallo
- Verificar último batch guardado
- El script continuará desde donde quedó si se reinicia
- Batches ya guardados NO se reescriben

---

**Implementado por:** Nicolás Schiaffino & Gabriel Venegas  
**Curso:** IIC3641 - Aprendizaje Basado en Grafos  
**Fecha:** 2025
