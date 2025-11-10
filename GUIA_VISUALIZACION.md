# Guía de Visualización de Grafos de Atención

## 📊 Descripción

El script `visualize_attention_graph.py` genera visualizaciones de los grafos de atención que se construyen a partir de los traces. Muestra exactamente cómo se ven los datos que alimentan el modelo de detección de alucinaciones.

## 🎨 Tipos de Visualizaciones

### 1. Grafo de Atención Individual
Visualiza una capa específica mostrando:
- **Nodos**: Tokens con sus etiquetas
- **Arcos**: Conexiones de atención (coloreadas por intensidad)
- **Información**: Prompt, respuesta y estadísticas

### 2. Comparación Entre Capas
Muestra cómo evoluciona el patrón de atención a través de las capas (primera, media, última).

### 3. Heatmap de Matriz de Atención
Visualización de la matriz completa de atención promediada sobre las cabezas.

## 🚀 Uso Básico

### Visualización Simple de Una Capa

```bash
python src/visualize_attention_graph.py \
    --data-pattern "traces_data/*.pkl" \
    --trace-idx 0 \
    --layer-idx 15 \
    --output-dir ./visualizations
```

**Resultado:** Un grafo de la capa 15 del primer trace.

### Visualización Completa (3 Tipos)

```bash
python src/visualize_attention_graph.py \
    --data-pattern "traces_data/*.pkl" \
    --trace-idx 0 \
    --layer-idx 15 \
    --compare-layers \
    --create-heatmap \
    --output-dir ./visualizations
```

**Resultado:** 
- `graph_layer_15.png` - Grafo individual
- `layerwise_comparison.png` - Comparación entre capas
- `attention_heatmap_layer_15.png` - Heatmap de atención

## 📋 Parámetros Completos

| Parámetro | Descripción | Default | Ejemplo |
|-----------|-------------|---------|---------|
| `--data-pattern` | Patrón glob para .pkl | **Requerido** | `"traces_data/*.pkl"` |
| `--trace-idx` | Índice del trace | `0` | `5` |
| `--layer-idx` | Capa a visualizar | `15` | `0`, `31` |
| `--attn-threshold` | Umbral para arcos | `0.01` | `0.05` |
| `--output-dir` | Directorio de salida | `./visualizations` | `./figuras` |
| `--layout` | Layout del grafo | `spring` | `circular`, `kamada_kawai` |
| `--max-nodes` | Máx. nodos a mostrar | `50` | `100` |
| `--compare-layers` | Comparación entre capas | `False` | Flag (activa) |
| `--create-heatmap` | Crear heatmap | `False` | Flag (activa) |

## 🎯 Ejemplos de Uso

### Ejemplo 1: Explorar Diferentes Capas

```bash
# Capa inicial (0)
python src/visualize_attention_graph.py \
    --data-pattern "traces_data/*.pkl" \
    --layer-idx 0 \
    --output-dir ./visualizations/layer_0

# Capa media (15)
python src/visualize_attention_graph.py \
    --data-pattern "traces_data/*.pkl" \
    --layer-idx 15 \
    --output-dir ./visualizations/layer_15

# Capa final (31)
python src/visualize_attention_graph.py \
    --data-pattern "traces_data/*.pkl" \
    --layer-idx 31 \
    --output-dir ./visualizations/layer_31
```

### Ejemplo 2: Diferentes Layouts

```bash
# Layout spring (default - distribuido)
python src/visualize_attention_graph.py \
    --data-pattern "traces_data/*.pkl" \
    --layout spring

# Layout circular (tokens en círculo)
python src/visualize_attention_graph.py \
    --data-pattern "traces_data/*.pkl" \
    --layout circular

# Layout secuencial (tokens en línea)
python src/visualize_attention_graph.py \
    --data-pattern "traces_data/*.pkl" \
    --layout sequential
```

### Ejemplo 3: Ajustar Threshold de Atención

```bash
# Threshold bajo (más arcos, más conexiones)
python src/visualize_attention_graph.py \
    --data-pattern "traces_data/*.pkl" \
    --attn-threshold 0.001

# Threshold alto (menos arcos, solo conexiones fuertes)
python src/visualize_attention_graph.py \
    --data-pattern "traces_data/*.pkl" \
    --attn-threshold 0.05
```

### Ejemplo 4: Visualizar Múltiples Traces

```bash
# Crear visualizaciones de los primeros 5 traces
for i in {0..4}; do
    python src/visualize_attention_graph.py \
        --data-pattern "traces_data/*.pkl" \
        --trace-idx $i \
        --layer-idx 15 \
        --compare-layers \
        --create-heatmap \
        --output-dir ./visualizations/trace_$i
done
```

## 🎨 Interpretación de las Visualizaciones

### Grafo de Atención

**Nodos (Tokens):**
- Color azul claro
- Etiqueta = texto del token
- Tamaño = 800 (fijo)

**Arcos (Atención):**
- Color: Azul claro → Azul → Naranja → Rojo
  - Azul claro: Atención débil
  - Rojo: Atención fuerte
- Dirección: Flecha indica `source → target`
- Grosor: Fijo (2.0)
- Solo se muestran arcos > threshold

**Colorbar:**
- Escala indica el rango de pesos de atención
- Min: Conexión más débil mostrada
- Max: Conexión más fuerte

### Comparación Entre Capas

Muestra 3 grafos lado a lado:
- **Capa 0** (inicial): Patrones de atención tempranos
- **Capa media** (15-16): Procesamiento intermedio  
- **Capa final** (31): Patrones de atención refinados

**Observaciones típicas:**
- Capas iniciales: Atención más dispersa
- Capas finales: Atención más concentrada en tokens relevantes

### Heatmap de Atención

**Ejes:**
- Eje X: Key tokens (a qué atiende)
- Eje Y: Query tokens (quién atiende)

**Colores:**
- Amarillo claro: Poca atención
- Naranja: Atención moderada
- Rojo oscuro: Atención alta

**Patrones comunes:**
- Diagonal: Auto-atención (token atiende a sí mismo)
- Bloques: Grupos de tokens relacionados
- Columnas destacadas: Tokens importantes (ej: palabras clave del prompt)

## 📊 Salida del Script

```
================================================================================
VISUALIZACIÓN DE GRAFOS DE ATENCIÓN
================================================================================

📂 Cargando trace 0 de traces_data/*.pkl...
✓ Trace cargado: qb_3343
  Número de capas: 32
  Tokens: 34
  Respuesta: Qatar....

🎨 Generando visualización de capa 15...
⚠️  El grafo tiene 34 nodos. Mostrando solo los primeros 50.
✅ Gráfico guardado en: visualizations/graph_layer_15.png

🎨 Generando comparación entre capas...
✅ Comparación guardada en: visualizations/layerwise_comparison.png

🎨 Generando heatmap de atención de capa 15...
✅ Heatmap guardado en: visualizations/attention_heatmap_layer_15.png

================================================================================
✅ VISUALIZACIÓN COMPLETADA
📁 Archivos guardados en: visualizations
================================================================================
```

## 🔧 Troubleshooting

### Problema: "No se encontraron archivos"
```bash
# Verificar el patrón
ls traces_data/*.pkl

# Ajustar patrón si es necesario
python src/visualize_attention_graph.py \
    --data-pattern "/ruta/completa/traces_data/*.pkl"
```

### Problema: "trace_idx fuera de rango"
```bash
# Ver cuántos traces hay
python -c "import pickle; print(len(pickle.load(open('traces_data/batch_0001.pkl', 'rb'))))"

# Usar índice válido
python src/visualize_attention_graph.py ... --trace-idx 0
```

### Problema: "Grafo muy grande"
```bash
# Reducir número de nodos mostrados
python src/visualize_attention_graph.py \
    ... \
    --max-nodes 30
```

### Problema: "Muy pocos arcos"
```bash
# Reducir threshold
python src/visualize_attention_graph.py \
    ... \
    --attn-threshold 0.001
```

## 💡 Consejos

1. **Para presentaciones:** Usar `--layout circular` (más limpio visualmente)

2. **Para análisis detallado:** Usar `--create-heatmap` (muestra patrones completos)

3. **Para comparaciones:** Usar `--compare-layers` (evolución a través de capas)

4. **Grafos grandes:** Combinar `--max-nodes 30` con `--attn-threshold 0.05`

5. **Alta resolución:** Las imágenes se guardan a 300 DPI, ideales para papers

## 📚 Dependencias

```bash
pip install networkx matplotlib numpy
```

## 🎯 Casos de Uso

### Para Papers/Presentaciones
```bash
python src/visualize_attention_graph.py \
    --data-pattern "traces_data/*.pkl" \
    --trace-idx 0 \
    --layer-idx 15 \
    --layout circular \
    --compare-layers \
    --create-heatmap \
    --max-nodes 40 \
    --output-dir ./paper_figures
```

### Para Debugging
```bash
python src/visualize_attention_graph.py \
    --data-pattern "traces_data/*.pkl" \
    --trace-idx 0 \
    --layer-idx 0 \
    --attn-threshold 0.001 \
    --output-dir ./debug
```

### Para Exploración
```bash
# Ver múltiples capas de un trace
for layer in 0 5 10 15 20 25 31; do
    python src/visualize_attention_graph.py \
        --data-pattern "traces_data/*.pkl" \
        --layer-idx $layer \
        --output-dir ./exploration/layer_$layer
done
```

---
**Tip:** Combina con `inspect_trace_structure.py` para primero entender tus datos, luego visualizarlos.
