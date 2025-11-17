# Divide and Conquer - Guía de Uso

## 📋 Descripción

Script para dividir batches grandes de traces (1000 traces, ~14GB) en batches más pequeños (250 traces, ~3.5GB cada uno) para mejorar la gestión de memoria.

## 🎯 Casos de Uso

### Problema
Tienes batches de 1000 traces que pesan ~14GB cada uno, lo cual:
- Consume demasiada RAM al cargar
- Hace el cache LRU menos eficiente
- Dificulta el procesamiento en máquinas con poca memoria

### Solución
Dividir cada batch de 1000 traces en 4 batches de 250 traces (~3.5GB cada uno)

## 🚀 Uso Rápido

### Dividir todos los archivos en un directorio
```bash
python divide_conquer.py \
    --input traces_data/ \
    --output traces_data_split/
```

Esto:
- Busca todos los archivos `.pkl` y `.pkl.gz` en `traces_data/`
- Divide cada uno en batches de 250 traces (default)
- Guarda los resultados en `traces_data_split/`
- Mantiene los archivos originales
- Valida que la división sea correcta

### Dividir un archivo específico
```bash
python divide_conquer.py \
    --input traces_data/llama2_chat_7B_triviaqa_batch_0000.pkl.gz \
    --output traces_data_split/
```

## ⚙️ Opciones Avanzadas

### Cambiar el tamaño de los batches
```bash
# Batches de 100 traces
python divide_conquer.py \
    --input traces_data/ \
    --output traces_data_split/ \
    --traces-per-batch 100

# Batches de 500 traces
python divide_conquer.py \
    --input traces_data/ \
    --output traces_data_split/ \
    --traces-per-batch 500
```

### Sin comprimir (archivos .pkl en lugar de .pkl.gz)
```bash
python divide_conquer.py \
    --input traces_data/ \
    --output traces_data_split/ \
    --no-compress
```

**Nota**: No recomendado. Los archivos sin comprimir ocupan ~3-4x más espacio.

### Eliminar archivos originales (¡CUIDADO!)
```bash
python divide_conquer.py \
    --input traces_data/ \
    --output traces_data_split/ \
    --delete-original
```

**⚠️ ADVERTENCIA**: 
- Solo elimina originales si la validación es exitosa
- Pedirá confirmación antes de proceder
- **Haz un backup antes de usar esta opción**

### Modo rápido (sin validación)
```bash
python divide_conquer.py \
    --input traces_data/ \
    --output traces_data_split/ \
    --no-validate
```

**Nota**: Más rápido pero menos seguro. Solo usar si confías en que el proceso funcionará bien.

## 📊 Ejemplo Completo

### Escenario
Tienes:
- `traces_data/batch_0000.pkl.gz` - 1000 traces, 14 GB
- `traces_data/batch_0001.pkl.gz` - 1000 traces, 14 GB

### Comando
```bash
python divide_conquer.py \
    --input traces_data/ \
    --output traces_data_split/ \
    --traces-per-batch 250
```

### Resultado
```
traces_data_split/
├── batch_0000_sub0000.pkl.gz  # Traces 0-249 (~3.5 GB)
├── batch_0000_sub0001.pkl.gz  # Traces 250-499 (~3.5 GB)
├── batch_0000_sub0002.pkl.gz  # Traces 500-749 (~3.5 GB)
├── batch_0000_sub0003.pkl.gz  # Traces 750-999 (~3.5 GB)
├── batch_0001_sub0000.pkl.gz  # Traces 0-249 (~3.5 GB)
├── batch_0001_sub0001.pkl.gz  # Traces 250-499 (~3.5 GB)
├── batch_0001_sub0002.pkl.gz  # Traces 500-749 (~3.5 GB)
└── batch_0001_sub0003.pkl.gz  # Traces 750-999 (~3.5 GB)
```

## 🔍 Proceso Detallado

Para cada archivo de entrada, el script:

1. **Carga** el archivo completo en memoria
2. **Divide** en sub-batches del tamaño especificado
3. **Guarda** cada sub-batch con nombre secuencial
4. **Valida** que el número total de traces coincide
5. **Libera** memoria con `gc.collect()` entre operaciones
6. **Reporta** el progreso y estadísticas

### Validación
El script valida automáticamente que:
- Número de traces original = suma de traces en sub-batches
- Todos los archivos se guardaron correctamente
- Si la validación falla, se aborta y se eliminan archivos parciales

## 💡 Recomendaciones

### Tamaño de Batch Óptimo

| RAM Disponible | Traces/Batch Recomendado | Archivos en Cache |
|----------------|--------------------------|-------------------|
| 8-16 GB        | 100-150                  | 2                 |
| 16-32 GB       | 200-250                  | 2-3               |
| 32-64 GB       | 250-500                  | 3-5               |
| 64+ GB         | 500-1000                 | 5-10              |

### Workflow Recomendado

1. **Hacer backup** de tus archivos originales
2. **Probar con un archivo** primero:
   ```bash
   python divide_conquer.py \
       --input traces_data/batch_0000.pkl.gz \
       --output traces_data_split_test/
   ```
3. **Verificar resultado** manualmente
4. **Procesar todo** si funciona:
   ```bash
   python divide_conquer.py \
       --input traces_data/ \
       --output traces_data_split/
   ```
5. **Probar con dataloader**:
   ```bash
   # Actualizar path en tu código
   dataset = TraceGraphDataset("traces_data_split/*.pkl.gz")
   ```
6. **Opcional**: Eliminar originales si todo funciona

## 🎯 Integración con Dataloader Optimizado

Después de dividir los batches, el dataloader optimizado será aún más eficiente:

```python
from dataloader import TraceGraphDataset

# Antes: cache de 2 archivos = 2 × 14GB = 28GB RAM
dataset = TraceGraphDataset("traces_data/*.pkl.gz")

# Después: cache de 2 archivos = 2 × 3.5GB = 7GB RAM
dataset = TraceGraphDataset("traces_data_split/*.pkl.gz")
```

**Beneficios adicionales**:
- Cache más eficiente (2 archivos de 3.5GB vs 14GB)
- Menor latencia al cargar archivos
- Mejor aprovechamiento de RAM
- Más archivos pueden caber en cache

## 📈 Estadísticas Esperadas

### Para 1 batch de 1000 traces (14GB)

**Antes**:
```
traces_data/batch_0000.pkl.gz  14 GB
```

**Después** (250 traces/batch):
```
traces_data_split/batch_0000_sub0000.pkl.gz  3.5 GB
traces_data_split/batch_0000_sub0001.pkl.gz  3.5 GB
traces_data_split/batch_0000_sub0002.pkl.gz  3.5 GB
traces_data_split/batch_0000_sub0003.pkl.gz  3.5 GB
─────────────────────────────────────────────────────
Total: 14 GB (mismo tamaño, 4 archivos)
```

### Tiempo Estimado

| Operación | Tiempo (aprox) |
|-----------|----------------|
| Cargar 14GB | 30-60 seg |
| Dividir en 4 | 10-20 seg |
| Guardar 4 archivos | 60-120 seg |
| Validar | 30-60 seg |
| **Total por archivo** | **2-4 min** |

Para 10 batches de 14GB: ~20-40 minutos

## 🐛 Troubleshooting

### Error: Out of Memory
**Problema**: No hay suficiente RAM para cargar el archivo completo

**Solución**:
```bash
# Procesar archivos uno a uno en lugar de todo el directorio
for file in traces_data/*.pkl.gz; do
    python divide_conquer.py --input "$file" --output traces_data_split/
    # Esperar que termine antes de procesar el siguiente
done
```

### Error: "No se pudo guardar"
**Problema**: No hay espacio en disco

**Solución**: 
- Verificar espacio: `df -h`
- Liberar espacio o usar otro directorio de salida
- Considerar `--delete-original` para liberar espacio progresivamente

### Archivos .part residuales
**Problema**: Quedan archivos `.part` de ejecuciones fallidas

**Solución**:
```bash
# Limpiar archivos .part
rm traces_data_split/*.part
```

### Validación falla
**Problema**: Número de traces no coincide

**Solución**:
- Verificar integridad del archivo original:
  ```python
  import pickle, gzip
  with gzip.open('archivo.pkl.gz', 'rb') as f:
      data = pickle.load(f)
      print(f"Traces: {len(data)}")
  ```
- Reportar el error con el archivo problemático

## 📝 Argumentos Completos

```
usage: divide_conquer.py [-h] --input INPUT --output OUTPUT
                        [--traces-per-batch N] [--no-compress]
                        [--delete-original] [--no-validate]
                        [--pattern PATTERN] [--quiet]

Argumentos:
  -h, --help            Mostrar ayuda
  --input, -i INPUT     Archivo o directorio de entrada (requerido)
  --output, -o OUTPUT   Directorio de salida (requerido)
  --traces-per-batch N, -n N
                        Traces por batch (default: 250)
  --no-compress         No comprimir salida con gzip
  --delete-original     Eliminar originales (¡CUIDADO!)
  --no-validate         No validar división
  --pattern PATTERN     Patrón de archivos (default: *.pkl*)
  --quiet, -q          Modo silencioso
```

## ✅ Checklist de Uso

Antes de ejecutar:
- [ ] Tienes backup de los datos originales
- [ ] Verificaste espacio en disco (≥ tamaño original)
- [ ] Probaste con un archivo primero
- [ ] Entiendes que `--delete-original` elimina los archivos

Durante ejecución:
- [ ] Monitorea uso de RAM: `watch free -h`
- [ ] Monitorea espacio en disco: `watch df -h`
- [ ] Revisa logs de errores

Después de ejecutar:
- [ ] Verifica que el número de archivos es correcto
- [ ] Prueba cargar archivos con dataloader
- [ ] Compara tamaños originales vs divididos
- [ ] Considera eliminar originales solo si todo funciona

## 🆘 Ayuda

Para ver ayuda completa:
```bash
python divide_conquer.py --help
```

Para reportar problemas, incluye:
- Comando ejecutado
- Output completo del script
- Tamaño de archivos: `ls -lh traces_data/`
- RAM disponible: `free -h`

---

**Última actualización**: 2024-11-17  
**Versión**: 1.0
