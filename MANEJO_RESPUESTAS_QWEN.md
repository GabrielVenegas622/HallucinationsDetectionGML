# Manejo Inteligente de Respuestas para Qwen-4B-Instruct

## Problema Identificado

Qwen-4B-Instruct tiene dificultades para generar el token EOS (End of Sentence) de manera consistente, lo que resulta en:
- **Redundancia**: El modelo continúa generando texto innecesario después de responder
- **Ruido en las trazas**: Se capturan activaciones de tokens irrelevantes
- **Desperdicio de recursos**: Procesamiento y almacenamiento innecesarios

---

## Solución Implementada

Sistema multi-estrategia para detectar y cortar automáticamente las respuestas en el punto óptimo.

### 1. Función: `find_answer_cutoff_point()`

Detecta el punto de corte usando 5 estrategias en orden de prioridad:

#### Estrategia 1: Primer punto (`.`)
```python
# Detecta el primer punto en la respuesta
"The answer is California." -> Corta en "California."
```
**Uso típico**: Respuestas concisas bien formadas

#### Estrategia 2: Primer salto de línea (`\n`)
```python
# Detecta cuando el modelo empieza a generar nuevo contenido
"California\nLet me explain..." -> Corta en "California"
```
**Uso típico**: Cuando el modelo agrega explicaciones no solicitadas

#### Estrategia 3: Signos de interrogación o exclamación (`?`, `!`)
```python
# Para preguntas retóricas o respuestas enfáticas
"What a great question!" -> Corta en "question!"
```
**Uso típico**: Respuestas expresivas

#### Estrategia 4: Detección de repetición
```python
# Detecta cuando el modelo se repite
"California California is..." -> Corta antes de la repetición
```
**Uso típico**: Generaciones redundantes o en bucle

#### Estrategia 5: Generación completa (fallback)
```python
# Si ninguna estrategia anterior funciona, usa toda la generación
```
**Uso típico**: Respuestas muy cortas o atípicas

---

## Mejoras en la Generación

### Parámetros Optimizados

```python
generation_output = model.generate(
    **prompt,
    num_beams=5,
    repetition_penalty=1.5,      # ↑ Aumentado de 1.2 a 1.5
    length_penalty=0.8,          # ★ NUEVO: Penaliza respuestas largas
    no_repeat_ngram_size=3,      # ★ NUEVO: Evita repetición de 3-gramas
    early_stopping=True,         # ★ NUEVO: Detiene en EOS si se genera
    eos_token_id=tokenizer.eos_token_id,  # ★ Configurado explícitamente
    ...
)
```

**Impacto esperado:**
- ✅ Menos redundancia (repetition_penalty + no_repeat_ngram_size)
- ✅ Respuestas más cortas (length_penalty)
- ✅ Mejor detección de fin (early_stopping + eos_token_id)

### Prompt Mejorado

**Antes:**
```python
prompt_text = f"Answer the question concisely. Q: {question} A:"
```

**Ahora:**
```python
prompt_text = f"Answer the question concisely in one sentence.\n\nQuestion: {question}\nAnswer:"
```

**Mejoras:**
- Instrucción explícita: "in one sentence"
- Formato más estructurado con saltos de línea
- Claridad mejorada para el modelo

---

## Datos Extraídos

### Nuevos Campos en Cada Trace

```python
{
    # Campos originales
    'question': str,
    'generated_text': str,
    'generated_answer': str,
    'hidden_states': list,
    'attentions': list,
    'tokens': np.ndarray,
    'prompt_length': int,
    'num_layers': int,
    
    # ★ NUEVOS campos
    'generated_answer_clean': str,      # Respuesta cortada en el punto óptimo
    'tokens_full': np.ndarray,          # Tokens completos (sin cortar)
    'cutoff_method': str,               # Método usado: 'first_period', 'repetition_detected', etc.
    'tokens_before_cutoff': int,        # Número de tokens antes del corte
    'tokens_after_cutoff': int          # Número de tokens descartados
}
```

### Importante: Las Trazas se Cortan

**Hidden states y attentions solo incluyen los tokens hasta el punto de corte:**

```python
# Si la respuesta es "California. California is a state..."
# Y se corta en "California."
# Entonces:
len(trace['hidden_states'][0])  # Solo tokens hasta el primer "."
len(trace['attentions'][0])     # Solo tokens hasta el primer "."
```

**Beneficios:**
- ✅ **Menos ruido**: Solo trazas relevantes
- ✅ **Menos almacenamiento**: ~30-50% menos espacio
- ✅ **Mejor calidad**: Grafos más limpios

---

## Estadísticas de Corte

Al finalizar la extracción, se muestran estadísticas:

```
📊 Estadísticas de métodos de corte:
   • first_period: 450 (90.0%)
   • first_newline: 35 (7.0%)
   • repetition_detected: 10 (2.0%)
   • question_mark: 3 (0.6%)
   • full_generation: 2 (0.4%)
```

Esto te permite monitorear:
- ¿Qué tan bien funciona cada estrategia?
- ¿Hay muchas respuestas con repetición?
- ¿El modelo genera EOS correctamente?

---

## Ejemplo de Uso

### Durante la Extracción

```bash
python src/trace_extractor.py
```

**Salida esperada:**
```
--- Ejemplo 10 (Batch actual: 11/500) ---
Pregunta: What U.S. state produces the most peaches?
Respuesta original: Georgia. Georgia is known for its peach production...
Respuesta limpia: Georgia.
Método de corte: first_period
Tokens usados: 3 (descartados: 8)
```

### En el Código

```python
from src.batch_loader import TraceBatchLoader

loader = TraceBatchLoader()
trace = loader.get_batch(0)[0]

print(f"Respuesta limpia: {trace['generated_answer_clean']}")
print(f"Método: {trace['cutoff_method']}")
print(f"Tokens útiles: {trace['tokens_before_cutoff']}")
print(f"Tokens descartados: {trace['tokens_after_cutoff']}")
```

---

## Configuración

### Activar/Desactivar Corte

En `extract_activations_and_attentions()`:

```python
traces = extract_activations_and_attentions(
    model=model,
    tokenizer=tokenizer,
    question=question,
    answer=answer_aliases,
    max_new_tokens=64,
    cut_at_period=True  # False para desactivar corte
)
```

### Ajustar Parámetros de Generación

En `src/trace_extractor.py`, líneas ~159-170:

```python
# Más conservador (respuestas más cortas)
repetition_penalty=2.0,    # Mayor penalización
length_penalty=0.5,        # Más agresivo
max_new_tokens=32,         # Límite más bajo

# Menos conservador (respuestas más largas)
repetition_penalty=1.2,
length_penalty=1.0,
max_new_tokens=128,
```

---

## Impacto en el Proyecto

### Beneficios para Detección de Alucinaciones

1. **Trazas más limpias**: Solo activaciones de la respuesta real
2. **Grafos más precisos**: Sin nodos de tokens redundantes
3. **Mejor entrenamiento**: Menos ruido en el VAE
4. **Comparaciones justas**: Todas las respuestas tienen longitudes comparables

### Ahorro de Recursos

**Estimación con corte en primer punto:**

| Métrica | Sin Corte | Con Corte | Ahorro |
|---------|-----------|-----------|--------|
| Tokens promedio | 40 | 15 | 62.5% |
| Tamaño por trace | 10 MB | 4 MB | 60% |
| Tamaño por batch (500) | 5 GB | 2 GB | 60% |
| Dataset completo (87k) | 870 GB | 350 GB | 60% |

**Nota**: Los porcentajes varían según la verbosidad del modelo en tu dataset específico.

---

## Validación

### Comprobar que Funciona

```bash
# Ejecutar prueba rápida
python src/test_quick.py
```

Observa la salida:
```
Respuesta original: Georgia. Georgia is a state that...
Respuesta limpia: Georgia.
Método: first_period
```

### Inspeccionar Resultados

```bash
python src/inspect_traces.py
```

Verifica:
- Distribución de métodos de corte
- Longitudes de respuestas limpias vs originales
- Tokens promedio descartados

---

## Troubleshooting

### Problema: Muchas respuestas usan "full_generation"

**Causa**: El modelo no genera puntos ni saltos de línea

**Solución**:
```python
# Ajustar el prompt para forzar formato
prompt_text = f"Answer in one short sentence ending with a period.\n\nQ: {question}\nA:"
```

### Problema: Se cortan respuestas válidas

**Causa**: La respuesta correcta tiene múltiples oraciones

**Solución**:
```python
# Modificar find_answer_cutoff_point() para buscar segundo punto
# O ajustar la lógica según tus necesidades
```

### Problema: Tokens descartados aún son muchos

**Causa**: El modelo es muy verboso

**Solución**:
```python
# Parámetros más agresivos
repetition_penalty=2.0,
length_penalty=0.5,
max_new_tokens=32,
```

---

## Recomendaciones

### Para Producción

1. **Ejecutar prueba con 100 ejemplos** primero
2. **Revisar estadísticas de corte** 
3. **Ajustar parámetros** según resultados
4. **Procesar dataset completo** una vez optimizado

### Para Análisis

1. **Comparar respuestas limpias vs originales** manualmente en ~20 ejemplos
2. **Verificar que no se pierda información crucial**
3. **Ajustar estrategias de corte** si es necesario

### Para el Paper/Proyecto

Mencionar en metodología:
- Estrategia de limpieza de respuestas
- Impacto en calidad de grafos
- Distribución de métodos de corte usados
- Ahorro de recursos logrado

---

## Resumen

✅ **5 estrategias de corte** automático  
✅ **Parámetros de generación optimizados**  
✅ **Prompt mejorado** para respuestas concisas  
✅ **Trazas solo de tokens relevantes**  
✅ **Estadísticas detalladas** de métodos usados  
✅ **60% de ahorro** estimado en almacenamiento  
✅ **Compatible con batching** existente  

**El sistema está listo para manejar las peculiaridades de Qwen-4B-Instruct de manera robusta y eficiente.**
