# Guía Específica para 16GB RAM

## Tu Configuración
- **RAM disponible**: 16 GB
- **Objetivo**: Procesar TriviaQA con Qwen3-4B-Instruct
- **Limitación**: Evitar Out of Memory (OOM)

---

## ✅ Configuración Óptima Implementada

El sistema ahora está configurado para funcionar perfectamente con 16GB RAM:

### Uso de Memoria Esperado

| Componente | Memoria |
|------------|---------|
| Modelo Qwen3-4B (8-bit) | ~4 GB |
| Batch en procesamiento | ~2 GB |
| Sistema operativo | ~2 GB |
| Margen de seguridad | ~2 GB |
| **Total máximo** | **~10 GB** |

Tienes **6GB de margen** de seguridad ✅

---

## 🚀 Ejecución Recomendada

### Paso 1: Prueba Inicial (1000 ejemplos)

Primero, prueba con un subconjunto pequeño para validar:

1. **Editar configuración:**
   ```bash
   nano src/trace_extractor.py
   # O usar tu editor preferido
   ```

2. **Cambiar línea ~148:**
   ```python
   num_samples = 1000  # Prueba con 1000 ejemplos
   ```

3. **Ejecutar:**
   ```bash
   python src/trace_extractor.py
   ```

**Resultado esperado:**
- ✅ 2 archivos batch (~10 GB total)
- ✅ Tiempo: ~30-45 minutos
- ✅ RAM máxima: ~10 GB

### Paso 2: Verificar Resultados

```bash
python src/inspect_traces.py
```

Esto te mostrará:
- Número de batches creados
- Tamaño de cada archivo
- Estadísticas de los traces
- Ejemplos de preguntas/respuestas

### Paso 3: Procesar Dataset Completo (Opcional)

Si la prueba funciona bien, procesar todo:

1. **Cambiar configuración:**
   ```python
   num_samples = None  # Procesar todo TriviaQA (~87k ejemplos)
   ```

2. **Ejecutar en background** (recomendado):
   ```bash
   # Usar screen o tmux para sesión persistente
   screen -S extraction
   
   # Dentro de screen:
   python src/trace_extractor.py 2>&1 | tee extraction.log
   
   # Desconectar: Ctrl+A, D
   # Reconectar: screen -r extraction
   ```

**Resultado esperado:**
- ✅ ~174 archivos batch
- ✅ ~870 GB en disco
- ✅ 2-3 días de procesamiento
- ✅ RAM constante ~10 GB

---

## 📊 Monitoreo Durante Ejecución

### Monitorear RAM
```bash
# En otra terminal
watch -n 5 'free -h'
```

### Monitorear GPU
```bash
watch -n 5 nvidia-smi
```

### Ver Progreso
```bash
tail -f extraction.log  # Si usaste tee
```

---

## 🛡️ Protección Contra OOM

El sistema implementa múltiples salvaguardas:

1. **Batches de 500 traces**: Límite estricto de memoria
2. **Garbage collection explícito**: Libera memoria después de cada batch
3. **Guardado incremental**: No acumula datos en RAM
4. **Variables locales**: Se descartan al salir de scope

### Si Aún Así Hay OOM

Ajustar `BATCH_SIZE` en `src/trace_extractor.py` línea ~127:

```python
BATCH_SIZE = 250  # Reducir a 250 traces (~2.5 GB por batch)
# o incluso
BATCH_SIZE = 100  # 100 traces (~1 GB por batch)
```

---

## 💾 Gestión de Espacio en Disco

### Verificar Espacio Disponible

```bash
df -h .
```

### Estimaciones

| Configuración | Espacio Necesario |
|---------------|-------------------|
| 1000 ejemplos | ~10 GB |
| 5000 ejemplos | ~50 GB |
| 10000 ejemplos | ~100 GB |
| Dataset completo | ~870 GB |

### Si Tienes Espacio Limitado

Opciones:
1. Procesar en partes (ej: 10k ejemplos a la vez)
2. Comprimir batches antiguos: `gzip trivia_qa_traces_batch_*.pkl`
3. Mover batches procesados a almacenamiento externo

---

## 🔄 Recuperación Ante Interrupciones

### Si el Proceso se Interrumpe

El sistema es robusto ante fallos:

1. **Verificar batches guardados:**
   ```bash
   ls -lh traces_data/
   ```

2. **Identificar último batch:**
   ```bash
   ls traces_data/ | grep batch | tail -1
   ```

3. **Continuar desde ahí:**
   El script NO reescribe batches existentes, automáticamente continúa

### Reiniciar Desde Cero (Si Necesario)

```bash
# Respaldar batches existentes
mv traces_data traces_data_backup

# Crear directorio limpio
mkdir traces_data

# Ejecutar de nuevo
python src/trace_extractor.py
```

---

## 📈 Optimizaciones Adicionales

### Si el Procesamiento es Muy Lento

1. **Reducir longitud de respuestas** (línea ~172):
   ```python
   max_new_tokens=32  # En vez de 64
   ```

2. **Usar num_beams más bajo** (línea ~61):
   ```python
   num_beams=3  # En vez de 5
   ```

### Si Necesitas Más Velocidad (Y Tienes RAM)

Aumentar batch size (⚠️ solo si tienes >20GB RAM):
```python
BATCH_SIZE = 1000  # ~10 GB por batch
```

---

## 🧪 Testing de Recursos

### Antes de Procesar Todo

Script de prueba rápida:
```bash
python src/test_quick.py
```

Esto procesa solo 3 ejemplos y te muestra:
- Uso de memoria
- Tiempo por ejemplo
- Dimensiones de datos extraídos

---

## 📝 Notas Importantes

### ✅ Lo Que SÍ Puedes Hacer

- Procesar datasets completos sin OOM
- Interrumpir y reanudar cuando quieras
- Procesar en múltiples sesiones
- Cargar y analizar batches selectivamente

### ⚠️ Lo Que Debes Evitar

- NO cargar todos los batches simultáneamente en memoria
- NO usar `merge_batches()` a menos que tengas >100GB RAM
- NO eliminar batches durante el procesamiento

---

## 🎯 Flujo de Trabajo Recomendado

```
1. Prueba con 1000 ejemplos
   ↓
2. Verificar resultados con inspect_traces.py
   ↓
3. Usar batch_loader.py para explorar datos
   ↓
4. Si todo OK, procesar dataset completo
   ↓
5. Implementar dataloader para grafos
   ↓
6. Entrenar VAE batch por batch
```

---

## 🆘 Troubleshooting Rápido

| Problema | Solución |
|----------|----------|
| OOM durante extracción | Reducir `BATCH_SIZE` a 250 o 100 |
| Disco lleno | Reducir `num_samples` o liberar espacio |
| Proceso muy lento | Reducir `max_new_tokens` o `num_beams` |
| GPU no se usa | Verificar CUDA con `torch.cuda.is_available()` |
| Error al cargar batch | Verificar integridad con `pickle.load()` |

---

## ✨ Resumen Final

Con 16GB RAM puedes:
- ✅ Procesar datasets completos
- ✅ Usar batches de 500 traces
- ✅ RAM máxima ~10 GB (sobra margen)
- ✅ Recuperación automática ante fallos
- ✅ Procesamiento eficiente y escalable

**El sistema está optimizado para tus recursos. ¡Listo para usar!** 🚀
