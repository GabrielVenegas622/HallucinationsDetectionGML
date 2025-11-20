# Fix para IterableDataset - Método __len__() Agregado

## Problema
El error `object type 'PreprocessedLSTMDataset' has no len()` ocurría porque las clases `IterableDataset` no tienen por defecto un método `__len__()`, pero este es requerido por algunas operaciones como:
- `random_split()` para dividir datasets
- Mostrar barras de progreso con `tqdm`
- Operaciones internas de PyTorch que asumen datasets con tamaño conocido

## Solución Implementada

Se agregó el método `__len__()` a ambas clases de dataset iterables:

### 1. PreprocessedLSTMDataset
```python
def __len__(self):
    """Retorna el número total de traces en el dataset"""
    return self.total_traces
```

### 2. PreprocessedGNNDataset
```python
def __len__(self):
    """Retorna el número total de traces en el dataset"""
    return self.total_traces
```

## Cómo Funciona

1. Durante la inicialización del dataset, se escanean todos los archivos `.pt` para contar el número total de traces
2. Este conteo se almacena en `self.total_traces`
3. El método `__len__()` simplemente retorna este valor pre-calculado
4. No hay overhead adicional ya que el escaneo ya se realizaba antes para reportar estadísticas

## Ventajas

- ✅ Permite usar `random_split()` y otras operaciones que requieren `len()`
- ✅ No afecta el rendimiento (el conteo ya se hacía antes)
- ✅ Mantiene la estrategia de lazy loading (solo se cargan archivos cuando se iteran)
- ✅ Compatible con múltiples workers

## Test Script

Se creó `test_preprocessing.py` para verificar que todo funciona correctamente:

```bash
python test_preprocessing.py --preprocessed-dir preprocessed_data
```

Este script:
1. Verifica que los archivos preprocesados existen
2. Prueba la carga con ambos datasets (LSTM y GNN)
3. Itera sobre muestras para verificar formatos
4. Verifica tipos de datos (dtype, shapes, etc.)
5. Prueba con DataLoader y batch processing

## Notas Importantes

- Los datasets siguen siendo `IterableDataset` (no Map-style)
- El lazy loading se mantiene: solo se carga 1 archivo por worker a la vez
- El `__len__()` es una aproximación del tamaño total, útil para progress bars
- Para shuffling se usa buffer local (no shuffling global perfecto, pero aceptable para datasets grandes)
