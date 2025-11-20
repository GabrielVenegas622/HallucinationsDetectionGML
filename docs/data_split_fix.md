# Fix: Corrección de la División Train/Val/Test

## Problema Identificado

Cuando se dividieron los archivos preprocessed con `divide_and_conquer.py`, la distribución de archivos entre train/val/test quedó incorrecta:
- **Train**: 30 archivos 
- **Val**: 7 archivos
- **Test**: 183 archivos ⚠️

Esto ocurría porque el código original en `baseline.py` usaba una sola variable `n_files` para ambos directorios (LSTM y GNN), lo que causaba problemas cuando las carpetas tenían diferente cantidad de archivos.

## Solución Implementada

Se modificó `baseline.py` (líneas ~1790-1810) para:

1. **Calcular el split independientemente** para cada tipo de datos:
   ```python
   # Usar el número de archivos de cada carpeta para el split
   n_lstm_files = len(lstm_files_shuffled)
   n_gnn_files = len(gnn_files_shuffled)
   
   # Split para LSTM
   train_split_lstm = int(0.7 * n_lstm_files)
   val_split_lstm = int(0.85 * n_lstm_files)
   
   train_lstm_files = lstm_files_shuffled[:train_split_lstm]
   val_lstm_files = lstm_files_shuffled[train_split_lstm:val_split_lstm]
   test_lstm_files = lstm_files_shuffled[val_split_lstm:]
   
   # Split para GNN
   train_split_gnn = int(0.7 * n_gnn_files)
   val_split_gnn = int(0.85 * n_gnn_files)
   
   train_gnn_files = gnn_files_shuffled[:train_split_gnn]
   val_gnn_files = gnn_files_shuffled[train_split_gnn:val_split_gnn]
   test_gnn_files = gnn_files_shuffled[val_split_gnn:]
   ```

2. **Leer dinámicamente** el número de archivos en cada directorio usando:
   ```python
   all_lstm_files = sorted(list(lstm_dir.glob('preprocessed_*.pt')))
   all_gnn_files = sorted(list(gnn_dir.glob('preprocessed_*.pt')))
   ```

## Resultados Esperados

Con 44 archivos totales (resultado de dividir 1 archivo en ~5 partes con `divide_and_conquer.py`):

| Dataset | Train | Val | Test | Train% | Val% | Test% |
|---------|-------|-----|------|--------|------|-------|
| **44 files** | 30 | 7 | 7 | 68.2% | 15.9% | 15.9% |

La distribución ahora es **balanceada** y respeta la proporción deseada de 70/15/15.

## Workflow Recomendado

1. **Preprocesar datos originales**:
   ```bash
   python src/preprocess_for_training.py \
       --data-pattern "traces_data/*.pkl*" \
       --scores-file ground_truth_scores.csv \
       --output-dir preprocessed_data
   ```

2. **Dividir archivos grandes** (si hay problemas de memoria):
   ```bash
   # Para LSTM
   python src/divide_and_conquer.py \
       --input-dir preprocessed_data/lstm_solo \
       --output-dir preprocessed_data/lstm_solo_split \
       --traces-per-part 50
   
   # Para GNN
   python src/divide_and_conquer.py \
       --input-dir preprocessed_data/gnn \
       --output-dir preprocessed_data/gnn_split \
       --traces-per-part 50
   ```

3. **Entrenar modelos** (el split se hace automáticamente):
   ```bash
   python src/baseline.py \
       --lstm-dir preprocessed_data/lstm_solo_split \
       --gnn-dir preprocessed_data/gnn_split \
       --batch-size 64 \
       --num-workers 4
   ```

## Scripts Relacionados

- **`baseline.py`**: Contiene la lógica de split corregida
- **`divide_and_conquer.py`**: Divide archivos grandes en chunks pequeños
- **`preprocess_for_training.py`**: Genera los archivos preprocessed iniciales

## Notas Importantes

- El split se hace a nivel de **archivos**, no de traces individuales
- Se usa `random.seed(42)` para reproducibilidad
- El shuffle es independiente para LSTM y GNN (cada uno tiene su propio orden aleatorio)
- Esta estrategia es compatible con `IterableDataset` y permite usar múltiples workers
