# Uso del Extractor de Trazas

## Descripción
El script `trace_extractor.py` extrae activaciones (hidden states) y atenciones de modelos de lenguaje para detectar alucinaciones.

## Flags disponibles

### `--model`
- **Descripción**: ID del modelo de HuggingFace a utilizar
- **Tipo**: string
- **Default**: `meta-llama/Llama-2-7b-chat-hf`
- **Ejemplos**:
  - `--model meta-llama/Llama-2-7b-chat-hf`
  - `--model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
  - `--model Qwen/Qwen2-4B-Instruct`

### `--dataset`
- **Descripción**: Dataset a utilizar para extraer trazas
- **Tipo**: string
- **Opciones**: `triviaqa` o `truthfulqa`
- **Default**: `triviaqa`
- **Ejemplo**:
  - `--dataset triviaqa`
  - `--dataset truthfulqa`

### `--num-samples`
- **Descripción**: Número de muestras a procesar del dataset
- **Tipo**: int
- **Default**: `None` (procesa todo el dataset)
- **Ejemplo**:
  - `--num-samples 1000` (procesa solo las primeras 1000 muestras)
  - `--num-samples 100` (útil para pruebas rápidas)

### `--cut-response`
- **Descripción**: Activa el corte inteligente de respuesta (corta en el primer punto/señal de fin)
- **Default**: Activado por defecto
- **Ejemplo**:
  - `--cut-response` (activado explícitamente, aunque es el default)

### `--no-cut-response`
- **Descripción**: Desactiva el corte de respuesta (usa la generación completa)
- **Ejemplo**:
  - `--no-cut-response` (desactiva el corte)

## Ejemplos de uso

### 1. Uso básico (configuración por defecto)
```bash
python trace_extractor.py
```
Esto usará:
- Modelo: `meta-llama/Llama-2-7b-chat-hf`
- Dataset: `triviaqa`
- Samples: Todas las muestras
- Corte: Activado

### 2. Extraer trazas de TruthfulQA con Llama-2
```bash
python trace_extractor.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset truthfulqa \
  --cut-response
```

### 3. Extraer solo 500 muestras para prueba
```bash
python trace_extractor.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset triviaqa \
  --num-samples 500
```

### 4. Usar Llama-3.1 pre-cuantizado sin corte
```bash
python trace_extractor.py \
  --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
  --dataset triviaqa \
  --no-cut-response
```

### 5. Extraer trazas de TruthfulQA con corte desactivado
```bash
python trace_extractor.py \
  --dataset truthfulqa \
  --no-cut-response \
  --num-samples 200
```

## Formato de archivos de salida

Los archivos se guardan en `./traces_data/` con el siguiente formato:

```
<modelo>_<dataset>_batch_<num_batch>.pkl
```

Ejemplos:
- `llama2_triviaqa_batch_0000.pkl`
- `llama2_triviaqa_batch_0001.pkl`
- `llama2_truthfulqa_batch_0000.pkl`
- `llama3.1_triviaqa_batch_0000.pkl`

Cada batch contiene hasta 500 trazas.

## Inspeccionar resultados

Para inspeccionar los archivos generados:

```bash
python inspect_traces.py
```

Este script:
- Detecta automáticamente qué dataset se usó
- Carga el dataset original para recuperar preguntas y respuestas
- Muestra 5 ejemplos por batch
- Proporciona estadísticas globales

## Estructura de cada traza

Cada traza guardada contiene:

```python
{
    'question_id': str,              # ID único de la pregunta
    'generated_answer_clean': str,   # Respuesta limpia generada
    'hidden_states': [               # Lista de arrays por capa
        np.array([seq_len, hidden_dim]),  # Capa 1
        np.array([seq_len, hidden_dim]),  # Capa 2
        ...
    ],
    'attentions': [                  # Lista de arrays por capa
        np.array([num_heads, seq_len, seq_len]),  # Capa 1
        np.array([num_heads, seq_len, seq_len]),  # Capa 2
        ...
    ],
    'tokens': np.array([...]),       # IDs de tokens (solo respuesta)
    'tokens_decoded': [str, ...]     # Tokens decodificados como strings
}
```

Donde:
- `seq_len` = longitud del prompt + longitud de la respuesta generada
- `hidden_dim` = dimensión oculta del modelo (ej: 4096 para Llama-2-7B)
- `num_heads` = número de cabezas de atención (ej: 32 para Llama-2-7B)

## Notas importantes

1. **Cuantización**: El script detecta automáticamente si el modelo está pre-cuantizado (contiene `bnb-4bit` o `bnb-8bit` en el nombre). Si no lo está, aplica cuantización de 4-bit automáticamente.

2. **Memoria**: Cada batch de 500 trazas ocupa aproximadamente 5 GB. El script guarda batches automáticamente para evitar quedarse sin memoria RAM.

3. **Corte inteligente**: Cuando está activado, el corte busca:
   - Primer punto (`.`)
   - Primer salto de línea (`\n`)
   - Primer `.\n`
   - Signos de interrogación (`?`) o exclamación (`!`)
   - Patrones de repetición
   - Si no encuentra ninguno, usa la generación completa

4. **Datasets**:
   - **TriviaQA**: Contiene campo `question_id` único
   - **TruthfulQA**: No tiene ID único, se usa `truthfulqa_<idx>`
