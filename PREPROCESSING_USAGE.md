# Uso del Script de Preprocesamiento

## Descripción

El script `preprocess_for_training.py` procesa los archivos `.pkl.gz` crudos y genera versiones optimizadas para cada modelo:

1. **LSTM-solo**: Secuencias de hidden states del último token por capa (125 MB por 250 muestras)
2. **GNN-det+LSTM y GVAE**: Grafos con atenciones promediadas por capa (5.3 GB por 250 muestras)

**Ventajas:**
- Reduce tiempo de carga de ~30 seg a <1 seg por batch
- Archivos procesados son más pequeños que los crudos
- El entrenamiento es 50-100× más rápido

## Uso Básico

```bash
conda run -n networks python src/preprocess_for_training.py \
    --data-pattern "traces_data/*.pkl*" \
    --scores-file traces_data/gt_llama2_chat_7B_improve.csv \
    --output-dir preprocessed_data \
    --attn-threshold 0.0 \
    --score-threshold 0.5
```

## Parámetros

- `--data-pattern`: Patrón glob para encontrar archivos .pkl o .pkl.gz
- `--scores-file`: Archivo CSV con scores BLEURT (debe tener columnas `question_id` y `bleurt_score`)
- `--output-dir`: Directorio donde guardar archivos procesados (default: `preprocessed_data`)
- `--attn-threshold`: Umbral mínimo de atención para crear arcos en grafos (default: 0.0)
- `--score-threshold`: Umbral de BLEURT para clasificar alucinaciones (default: 0.5)

## Estructura de Salida

```
preprocessed_data/
├── lstm_solo/
│   ├── batch_0000.pt
│   ├── batch_0001.pt
│   └── ...
└── gnn/  # Para GNN-det+LSTM y GVAE
    ├── batch_0000.pt
    ├── batch_0001.pt
    └── ...
```

Cada archivo `.pt` contiene:

### LSTM-solo (`lstm_solo/batch_XXXX.pt`)
```python
{
    'sequences': Tensor[batch_size, num_layers, hidden_dim],
    'labels': Tensor[batch_size],
    'question_ids': List[str]
}
```

### GNN (`gnn/batch_XXXX.pt`)
```python
{
    'graphs': List[List[PyG.Data]],  # [batch_size][num_layers]
    'labels': Tensor[batch_size],
    'question_ids': List[str]
}
```

Donde cada grafo PyG.Data tiene:
- `x`: Tensor[num_nodes, hidden_dim] - Hidden states
- `edge_index`: Tensor[2, num_edges] - Conectividad del grafo
- `edge_attr`: Tensor[num_edges] - Valores de atención promediados
- `num_nodes`: int - Número de nodos (tokens)

## Notas

- GNN-det+LSTM y GVAE comparten la misma estructura de grafos
- Los números de batch se extraen automáticamente de los nombres de archivo
- Si un archivo falla, se omite y se continúa con el siguiente
- El proceso libera memoria automáticamente después de cada batch
