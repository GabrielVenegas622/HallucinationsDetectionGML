# Extractor de Trazas para Detección de Alucinaciones

## Descripción General

Este script (`trace_extractor.py`) implementa la extracción de activaciones (hidden states) y matrices de atención de todas las capas del modelo **Qwen3-4B-Instruct** durante la generación de respuestas a preguntas del dataset **TriviaQA**.

## Características Principales

### 1. Modelo Utilizado
- **Modelo**: `Qwen/Qwen3-4B-Instruct-2507`
- **Cuantización**: 8-bit usando BitsAndBytes
- **Implementación de atención**: `eager` (necesario para capturar matrices de atención)

### 2. Dataset
- **Dataset**: TriviaQA (`mandarjoshi/trivia_qa`, configuración `rc.nocontext`)
- **Split**: `train`
- Por defecto procesa 100 muestras (configurable en `num_samples`)

### 3. Datos Extraídos

Para cada pregunta del dataset, el script extrae:

#### Hidden States (Activaciones)
- Estructura: `[num_layers][num_tokens_generated][batch, seq_len, hidden_dim]`
- Contiene las activaciones de cada capa para cada token generado
- Se excluyen los embeddings iniciales

#### Attention Matrices (Atenciones)
- Estructura: `[num_layers][num_tokens_generated][batch, num_heads, seq_len, seq_len]`
- Matrices de atención completas de cada cabeza en cada capa
- Fundamentales para construir grafos de atención

#### Metadata
- Pregunta original
- Respuesta generada (completa y sin prompt)
- Tokens generados (IDs)
- Respuestas ground truth (aliases de TriviaQA)
- Longitud del prompt
- Número de capas

## Uso

### Requisitos Previos

```bash
pip install torch transformers datasets bitsandbytes accelerate tqdm
```

### Ejecución

```bash
cd /home/gaara/mnt/USM/2025-02/IIC3641/HallucinationsDetectionGML
python src/trace_extractor.py
```

### Configuración

Puedes ajustar los siguientes parámetros en `main()`:

```python
BATCH_SIZE = 500      # Traces por archivo (500 ≈ 5GB)
num_samples = None    # None = todo el dataset, o un número específico
max_new_tokens = 64   # Tokens máximos por respuesta
```

**Estimación de espacio en disco:**
- 1 trace ≈ 10 MB (varía según longitud de respuesta)
- 500 traces ≈ 5 GB por batch
- Dataset completo TriviaQA (~87k ejemplos) ≈ 870 GB

**Recomendación:** Empieza con `num_samples = 1000` para probar.

## Salida

### Archivos Generados (Modo Batch)

Los datos se guardan en archivos separados cada 500 traces:
```
./traces_data/
├── trivia_qa_traces_batch_0000.pkl  # Traces 0-499
├── trivia_qa_traces_batch_0001.pkl  # Traces 500-999
├── trivia_qa_traces_batch_0002.pkl  # Traces 1000-1499
└── ...
```

**Ventajas del modo batch:**
- ✅ Uso eficiente de memoria (máx ~5GB por batch)
- ✅ Procesamiento paralelo posible
- ✅ Recuperación ante fallos (batches ya guardados se conservan)
- ✅ Carga selectiva de datos

### Estructura del Pickle (Por Batch)

Cada batch contiene una lista de diccionarios, donde cada elemento es:

```python
{
    'question': str,                      # Pregunta original
    'generated_text': str,                # Texto generado completo
    'generated_answer': str,              # Respuesta sin prompt
    'hidden_states': list,                # [num_layers][num_tokens]
    'attentions': list,                   # [num_layers][num_tokens]
    'tokens': np.ndarray,                 # IDs de tokens generados
    'prompt_length': int,                 # Longitud del prompt
    'num_layers': int,                    # Número de capas
    'example_id': int,                    # ID dentro del batch (0-499)
    'global_example_id': int,             # ID global en el dataset
    'batch_number': int,                  # Número del batch
    'ground_truth_answers': list          # Respuestas correctas
}
```

### Trabajar con Batches

```python
from src.batch_loader import TraceBatchLoader

# Inicializar loader
loader = TraceBatchLoader("./traces_data")

# Iterar sobre todos los traces sin cargar todo en memoria
for trace in loader.iter_traces():
    print(trace['question'])

# Cargar un batch específico
batch_0 = loader.get_batch(0)  # Carga solo 500 traces

# Obtener información sin cargar datos
info = loader.get_batch_info()
```

Ver `src/batch_loader.py` para más opciones.

## Aplicación al Proyecto

### Construcción de Grafos

Las matrices de atención extraídas pueden usarse directamente para:

1. **Crear grafos por capa**: Donde cada token es un nodo y las atenciones son pesos de aristas
2. **Usar activaciones como features**: Los hidden states pueden ser features de los nodos

### Pipeline Siguiente

Según la presentación del proyecto, los siguientes pasos son:

1. **Procesamiento de Grafos**: Transformar matrices de atención en grafos `G_l` por capa
2. **VAE para Grafos**: Entrenar un Variational Autoencoder para obtener representaciones latentes `z_l`
3. **Detección de Anomalías**: Usar MLP para detectar cambios estructurales entre capas consecutivas

## Notas Técnicas

### Por qué `attn_implementation="eager"`

El modo "eager" es necesario para capturar las matrices de atención completas. Otras implementaciones optimizadas (flash attention) no retornan las matrices explícitamente.

### Gestión de Memoria

- Se usa cuantización de 8-bit para reducir uso de memoria
- Los datos se mueven a CPU inmediatamente después de la extracción
- Se guarda en disco en formato pickle para evitar problemas de memoria

### Reproducibilidad

- Semilla fijada en `SEED_VALUE = 41`
- `do_sample=False` para generación determinística
- `num_beams=5` para beam search consistente

## Próximos Pasos

1. **Implementar dataloader**: Cargar los grafos guardados eficientemente
2. **Construir grafos**: Convertir matrices de atención en estructuras de grafos (PyTorch Geometric)
3. **Implementar VAE**: Entrenar sobre los grafos para obtener embeddings latentes
4. **Scoring de alucinaciones**: MLP sobre secuencias de `z_l`

## Troubleshooting

### Error de memoria

Reduce `num_samples` o `max_new_tokens`:
```python
num_samples = 50
max_new_tokens = 32
```

### Token de acceso

Si el modelo requiere autenticación, coloca tu token en:
```
llama_token.txt
```

## Referencias

Este código implementa la fase de "Avance de Proyecto" según la presentación:
- Actividad 2: Script de "autopsia" para extraer hidden states y scores de atención
- Actividad 3: Guardar dataset de secuencias de grafos en disco
