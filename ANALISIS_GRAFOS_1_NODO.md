# Análisis: Grafos de 1 Nodo vs Atención al Prompt

## 🤔 Tu Pregunta

> "Me parece curioso que algunos grafos solo tengan un token ya que también quiero modelar su atención con respecto al prompt que, evidentemente, no tendrá un solo token. ¿Es esto manejado por el dataloader?"

**Respuesta corta:** Tienes razón en preocuparte. Si encuentras grafos de 1 nodo, es una señal de alerta.

## 🔍 Qué Debería Estar Pasando

Según el `trace_extractor.py`, los traces SE EXTRAEN con **prompt + respuesta completa**:

```python
# De trace_extractor.py líneas 111-154
seq_len_total = prompt_length + num_tokens_generated

# hidden_states: [seq_len_total, hidden_dim] 
# Incluye TODOS los tokens (prompt + respuesta)

# attentions: [num_heads, seq_len_total, seq_len_total]
# Matriz completa que incluye atención prompt↔respuesta
```

Por lo tanto, **cada grafo DEBERÍA tener**:
- Nodos = prompt tokens + respuesta tokens
- Atenciones entre TODOS los nodos (prompt puede atender a respuesta y viceversa)

## ❓ Entonces ¿Por Qué Aparecen Grafos de 1 Nodo?

Hay 3 posibles escenarios:

### Escenario 1: Bug en la Extracción ❌
```python
# Si solo se guardó la respuesta, no el prompt:
hidden_states = solo_respuesta  # Shape: (1, 4096) para respuesta de 1 token
attentions = matriz_completa    # Shape: (32, 512, 512) incluyendo prompt
```

**Diagnóstico:** Mismatch grande (1 vs 512)
**Solución:** Re-extraer traces correctamente

### Escenario 2: Respuestas Realmente Cortas ✓
```python
# Algunas preguntas tienen respuestas de 1 token + prompt normal:
hidden_states = (prompt + 1_token_respuesta)  # Shape: (50, 4096)
attentions = matriz_completa                   # Shape: (32, 50, 50)
```

**Diagnóstico:** Dimensiones coinciden (50 vs 50)
**Solución:** No hay problema, es normal

### Escenario 3: Padding en Extracción ⚠️
```python
# Se extrajo con max_length fijo pero seq_len real varía:
hidden_states = secuencia_real   # Shape: (actual_len, 4096)
attentions = matriz_paddeada     # Shape: (32, max_len, max_len)
```

**Diagnóstico:** attn más grande que hidden_states
**Solución:** El dataloader recorta automáticamente ✓

## 🚀 Cómo Verificar Tu Caso

### Paso 1: Inspeccionar Estructura
```bash
python src/inspect_trace_structure.py \
    --data-pattern "traces_data/*.pkl" \
    --num-samples 5
```

**Esto te dirá:**
- Cuántos nodos tienen tus grafos
- Si hidden_states y attentions coinciden
- Si hay grafos sospechosos de 1 nodo

### Paso 2: Interpretar Resultados

**Caso A: Todo coincide**
```
Capa 0: hidden_states=45 tokens, attentions=45x45 ✓
✅ TODAS LAS CAPAS tienen dimensiones consistentes!
Prompt estimado: ~35 tokens
Respuesta estimada: ~10 tokens
```
→ **Perfecto!** Los grafos incluyen prompt + respuesta

**Caso B: Grafos de 1 nodo con atenciones grandes**
```
⚠️ Capa 0: hidden_states=1 tokens, attentions=512x512
❌ CRÍTICO: Solo 1 nodo pero atenciones grandes.
¿Se guardó solo la respuesta y no el prompt?
```
→ **Problema!** Solo se guardó la respuesta, no el prompt

**Caso C: Atenciones más grandes (padding)**
```
Capa 0: hidden_states=45 tokens, attentions=512x512
⚠️ WARNING: Atenciones >> hidden_states. Recortando automáticamente.
```
→ **OK con corrección:** El dataloader recorta a 45x45

## ✅ Qué Hace el Dataloader

El dataloader actualizado:

1. **Detecta** el número real de nodos desde `hidden_states`
2. **Valida** que coincida con attentions
3. **Recorta** attentions si es más grande
4. **Muestra warnings** si detecta situaciones sospechosas
5. **Filtra** índices fuera de rango

```python
# Si num_nodes=45 pero attn_avg es 512x512:
attn_avg = attn_avg[:num_nodes, :num_nodes]  # Recorta a 45x45

# Si encuentra nodo=1 y attn=512x512:
print("⚠️ CRÍTICO: Solo 1 nodo pero atenciones grandes.")
```

## 🎯 Qué Deberías Ver en los Warnings

Al cargar el dataset con el dataloader actualizado, verás:

**Si todo está bien:**
```
Dataset secuencial creado:
  - 1000 traces
  - 32 capas por trace
```
(Sin warnings)

**Si hay problemas:**
```
⚠️ WARNING: Trace qid_123, capa 0: Atenciones (512x512) >> hidden_states (1 nodos)
   ⚠️ CRÍTICO: Solo 1 nodo pero atenciones grandes. 
   ¿Se guardó solo la respuesta y no el prompt?
```

## 🔧 Si Encuentras el Problema

### Solución Temporal (Dataloader lo Maneja)
El código actual funciona automáticamente, pero **pierdes información** de las atenciones al prompt.

### Solución Correcta (Re-extraer)
Si confirmas que solo se guardó la respuesta:

1. **Re-ejecutar trace_extractor.py** asegurando que se guarde todo:
   ```python
   # Verificar en trace_extractor.py línea 125:
   final_state_full = final_state[0, :seq_len_total, :].cpu().numpy()
   # seq_len_total DEBE incluir prompt + respuesta
   ```

2. **Validar** con inspect_trace_structure.py

3. **Entrenar** con traces correctos

## 📊 Importancia para el Modelo

**¿Por qué importa incluir el prompt?**

1. **Detección de alucinaciones:** Las atenciones del modelo hacia el contexto (prompt) son cruciales para detectar cuándo el modelo "inventa" información no presente en el input.

2. **Análisis de dependencias:** Un grafo con solo la respuesta pierde información sobre:
   - Qué partes del prompt influyeron en cada token
   - Si el modelo atendió al contexto relevante
   - Patrones de atención anómalos que indican alucinación

3. **Estructura completa:** El grafo debe modelar:
   ```
   prompt_tokens → atención → response_tokens
   response_tokens → atención → prompt_tokens
   response_tokens → atención → response_tokens
   ```

## ✅ Acción Recomendada

```bash
# 1. Inspeccionar tus traces
python src/inspect_trace_structure.py --data-pattern "traces_data/*.pkl"

# 2. Si muestra warnings críticos de "1 nodo":
#    → Re-extraer traces con trace_extractor.py

# 3. Si solo hay warnings de padding (attn > hidden_states):
#    → Continuar normalmente, el dataloader lo maneja

# 4. Validar que funciona
python src/quick_test.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv

# 5. Entrenar
python src/baseline.py \
    --data-pattern "traces_data/*.pkl" \
    --scores-file ground_truth_scores.csv \
    --batch-size 16 \
    --epochs 50
```

## 📝 Resumen

| Situación | Qué Significa | Acción |
|-----------|---------------|--------|
| Grafos 40-100 nodos | ✓ Incluye prompt + respuesta | Perfecto, entrenar |
| Grafos 1-5 nodos con attn grandes | ❌ Solo respuesta, no prompt | Re-extraer traces |
| Attn > hidden_states (padding) | ⚠️ Extracción con padding | OK, dataloader corrige |
| Attn = hidden_states | ✓ Consistente | Perfecto, entrenar |

---
**Última actualización:** 2024-11-09
**Conclusión:** El dataloader maneja el problema técnicamente, pero **debes verificar** que tus grafos incluyan el prompt para un análisis completo de alucinaciones.
