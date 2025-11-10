# Comandos de Referencia Rápida

## 🧪 Verificación de Implementación

```bash
# Test de compresión
python src/test_compression.py

# Debe mostrar: ✅ TODOS LOS TESTS PASARON
```

## 📦 Re-Extracción de Traces

```bash
# Extracción completa (1000 traces)
python src/trace_extractor.py \
    --model-id llama2_chat_7B \
    --dataset triviaqa \
    --num-samples 1000

# Archivos generados: traces_data/*.pkl.gz (~1.5-2.5 GB total)
```

## ✅ Validación de Traces

```bash
# Validar dimensiones
python src/validate_traces.py --data-pattern "traces_data/*.pkl*"

# Inspeccionar estructura
python src/inspect_trace_structure.py --data-pattern "traces_data/*.pkl*"

# Visualizar grafo
python src/visualize_attention_graph.py \
    --data-pattern "traces_data/*.pkl*" \
    --trace-idx 0 \
    --layer-idx 15 \
    --compare-layers \
    --create-heatmap
```

## 🧠 Entrenamiento

```bash
# Quick test
python src/quick_test.py \
    --data-pattern "traces_data/*.pkl*" \
    --scores-file ground_truth_scores.csv

# Entrenamiento completo
python src/baseline.py \
    --data-pattern "traces_data/*.pkl*" \
    --scores-file ground_truth_scores.csv \
    --batch-size 16 \
    --epochs 50 \
    --score-threshold 0.5
```

## 🔍 Verificación de Archivos

```bash
# Ver tamaño de archivos
ls -lh traces_data/*.pkl*

# Contar traces en un archivo
python -c "import pickle, gzip; f=gzip.open('traces_data/batch_0001.pkl.gz','rb'); print(len(pickle.load(f)))"

# Verificar dtype
python -c "import pickle, gzip; f=gzip.open('traces_data/batch_0001.pkl.gz','rb'); t=pickle.load(f); print(f'HS dtype: {t[0][\"hidden_states\"][0].dtype}, Attn dtype: {t[0][\"attentions\"][0].dtype}')"
```

## 🛠️ Troubleshooting

```bash
# Si no encuentra archivos
--data-pattern "traces_data/*.pkl*"  # Incluye .pkl y .pkl.gz

# Test rápido en CPU
python src/baseline.py ... --force-cpu --batch-size 8

# Diagnóstico de errores
python src/diagnose_cuda_error.py \
    --data-pattern "traces_data/*.pkl*" \
    --scores-file ground_truth_scores.csv
```

---
**Versión:** 2.4
**Última actualización:** 2024-11-09
