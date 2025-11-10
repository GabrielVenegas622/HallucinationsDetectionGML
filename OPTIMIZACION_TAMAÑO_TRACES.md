# Optimización de Tamaño de Traces

## Problema
Los traces correctos ocupan ~15 GB por cada 1000 muestras (vs 350 MB con el bug).

## Solución Recomendada: Gzip + Float16

### Modificar trace_extractor.py

```python
import gzip
import pickle
import numpy as np

def extract_activations_and_attentions(model, tokenizer, question, max_new_tokens=64):
    """... código existente ..."""
    
    # Al final, antes del return:
    
    # Convertir a float16 para ahorrar espacio (pérdida mínima de precisión)
    hidden_states_by_layer = [hs.astype(np.float16) for hs in hidden_states_by_layer]
    attentions_by_layer = [attn.astype(np.float16) for attn in attentions_by_layer]
    
    return {
        'hidden_states': hidden_states_by_layer,
        'attentions': attentions_by_layer,
        'tokens': full_sequence.cpu().numpy(),
        'tokens_decoded': tokens_decoded,
        'generated_answer_clean': generated_answer_clean
    }

# Al guardar batch (línea ~280):
output_file = output_dir / f"{model_name}_{dataset_name}_batch_{batch_num:04d}.pkl.gz"

# Usar gzip
with gzip.open(output_file, 'wb', compresslevel=6) as f:
    pickle.dump(batch_data, f)
```

### Modificar dataloader.py

```python
import gzip

class TraceGraphDataset(TorchDataset):
    def __init__(self, pkl_files_pattern, attn_threshold=0.01):
        # ... código existente ...
        
        file_paths = glob.glob(pkl_files_pattern)
        
        for file_path in tqdm(file_paths, desc="Cargando traces"):
            # Detectar si es gzip
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rb') as f:
                    batch_data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    batch_data = pickle.load(f)
            
            self.all_traces.extend(batch_data)
```

## Resultados Esperados

| Archivo | Sin Optimizar | Con Gzip | Con Float16 | Gzip + Float16 |
|---------|---------------|----------|-------------|----------------|
| 1000 traces | 15 GB | 3-4 GB | 7.5 GB | **1.5-2.5 GB** |
| 5000 traces | 75 GB | 15-20 GB | 37.5 GB | **7.5-12.5 GB** |

## Implementación Rápida

```bash
# 1. Actualizar trace_extractor.py con los cambios
# 2. Re-extraer (solo 1 batch para probar)
python src/trace_extractor.py \
    --model-id llama2_chat_7B \
    --dataset triviaqa \
    --num-samples 100

# 3. Verificar tamaño
ls -lh traces_data/*.pkl.gz

# Deberías ver: ~150 MB para 100 traces (vs 1.5 GB sin optimizar)
```

## Pros y Contras

### Gzip
✅ Sin pérdida de información
✅ Reducción 70-80%
✅ Fácil de implementar
⚠️ Ligeramente más lento al cargar (no significativo)

### Float16
✅ Reducción 50%
✅ Más rápido que gzip
✅ Pérdida mínima de precisión (< 0.1%)
⚠️ Rango más limitado (puede overflow en valores extremos)

### Combinados
✅ Mejor reducción (85-90%)
✅ Sin pérdida significativa
✅ Balance ideal espacio/velocidad
⚠️ Require ambos cambios

## Validación Post-Optimización

```bash
# Después de re-extraer con optimización:
python src/inspect_trace_structure.py \
    --data-pattern "traces_data/*.pkl.gz"

# Debe mostrar:
# Hidden states: shape=(34, 4096), dtype=float16 ✓
# Attentions: shape=(32, 34, 34), dtype=float16 ✓
```

## Si Necesitas Aún Más Compresión

### Opción Avanzada: Guardar Solo Capas Relevantes

```python
# En trace_extractor.py
LAYERS_TO_SAVE = [0, 5, 10, 15, 20, 25, 31]  # 7 capas en lugar de 32

hidden_states_by_layer = []
for layer_idx in LAYERS_TO_SAVE:
    hs = outputs.hidden_states[layer_idx + 1][0].cpu().numpy().astype(np.float16)
    hidden_states_by_layer.append(hs)

# Reducción adicional: 2.5 GB → 0.6 GB por 1000 traces
```

Pero esto limita tu análisis a esas capas específicas.

## Conclusión

Para 1000 traces:
- **Sin optimizar:** 15 GB (correcto pero grande)
- **Recomendado:** 1.5-2.5 GB (Gzip + Float16)
- **Mínimo posible:** 0.6 GB (Gzip + Float16 + Capas selectivas)

El aumento de 350 MB → 15 GB es **esperado y correcto**.
Usa Gzip + Float16 para reducir a 1.5-2.5 GB sin pérdida significativa.
