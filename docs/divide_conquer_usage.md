# Guía de Uso: divide_and_conquer.py

## Propósito

Script para dividir archivos `.pt` grandes (con 250 traces) en partes más pequeñas (50 traces cada una). Esto optimiza el uso de memoria RAM durante el entrenamiento con DataLoader y múltiples workers.

## Contexto

Los archivos preprocessados actuales contienen 250 traces cada uno (~2.6GB para GNN). Esto causa problemas de memoria cuando se intenta usar múltiples workers en el DataLoader. Al dividirlos en partes más pequeñas de 50 traces, cada worker puede cargar archivos independientes sin saturar la RAM.

## Uso Básico

```bash
# Dividir archivos LSTM
python src/divide_and_conquer.py \
    --input-dir preprocessed_data/lstm_solo \
    --output-dir preprocessed_data/lstm_solo_divided

# Dividir archivos GNN
python src/divide_and_conquer.py \
    --input-dir preprocessed_data/gnn \
    --output-dir preprocessed_data/gnn_divided

# Especificar traces por parte (default: 50)
python src/divide_and_conquer.py \
    --input-dir preprocessed_data/gnn \
    --output-dir preprocessed_data/gnn_divided \
    --traces-per-part 25
```

## Estructura de Entrada/Salida

### Entrada
Archivo: `preprocessed_llama2_chat_7B_triviaqa_batch_0000_sub0000.pt`

Estructura del diccionario:
```python
{
    'graphs': [Data_0, Data_1, ..., Data_249],  # Lista de 250 objetos PyG Data
    'labels': Tensor([0, 1, ..., 1]),           # Tensor de 250 etiquetas
    'question_ids': ['id_0', 'id_1', ..., 'id_249']  # Lista de 250 IDs
}
```

### Salida
Se generan 5 archivos (250 traces / 50 traces por parte = 5):
- `preprocessed_llama2_chat_7B_triviaqa_batch_0000_sub0000_part0.pt` (traces 0-49)
- `preprocessed_llama2_chat_7B_triviaqa_batch_0000_sub0000_part1.pt` (traces 50-99)
- `preprocessed_llama2_chat_7B_triviaqa_batch_0000_sub0000_part2.pt` (traces 100-149)
- `preprocessed_llama2_chat_7B_triviaqa_batch_0000_sub0000_part3.pt` (traces 150-199)
- `preprocessed_llama2_chat_7B_triviaqa_batch_0000_sub0000_part4.pt` (traces 200-249)

Cada archivo contiene:
```python
{
    'graphs': [Data_i, ..., Data_j],  # 50 grafos
    'labels': Tensor([...]),           # 50 etiquetas
    'question_ids': ['id_i', ..., 'id_j']  # 50 IDs
}
```

## Ventajas

1. **Paralelización Real**: Múltiples workers pueden cargar archivos diferentes simultáneamente
2. **Menor uso de RAM**: Cada worker carga solo 50 traces (~260MB) en lugar de 250 (~1.3GB)
3. **Velocidad**: El entrenamiento es mucho más rápido con workers=4-8

## Después de Dividir

Actualiza el argumento en baseline.py:

```bash
# Antes (archivos grandes)
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --epochs 50

# Después (archivos divididos)
python src/baseline.py \
    --preprocessed-dir preprocessed_data/lstm_solo_divided \
    --preprocessed-gnn-dir preprocessed_data/gnn_divided \
    --epochs 50
```

## Notas

- El script libera memoria después de procesar cada archivo
- Usa `tqdm` para mostrar progreso
- Maneja errores por archivo (si uno falla, continúa con el siguiente)
- Preserva la sincronización entre `graphs`, `labels` y `question_ids`
