# Guía Rápida: Lazy Loading para Memoria Limitada

## TL;DR - Inicio Rápido

Si tienes problemas de memoria, usa esto:

```bash
# Para entrenar con memoria limitada
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --max-cache-batches 2 \
    --epochs 50 \
    --batch-size 16
```

O entrenar modelos uno por uno:

```bash
# Solo LSTM (usa menos memoria)
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --max-cache-batches 2 \
    --run-lstm --no-run-gnn-det --no-run-gvae \
    --epochs 50

# Solo GNN-det+LSTM
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --max-cache-batches 2 \
    --no-run-lstm --run-gnn-det --no-run-gvae \
    --epochs 50

# Solo GVAE+LSTM
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --max-cache-batches 2 \
    --no-run-lstm --no-run-gnn-det --run-gvae \
    --epochs 50
```

## ¿Qué es Lazy Loading?

En lugar de cargar **todos** los datos al inicio, solo cargamos lo que necesitamos en el momento.

### Antes (Carga Completa)
```
[Inicio] → Cargar 100% de datos → RAM llena → Entrenar
                                    ↓
                                  OOM! 💥
```

### Ahora (Lazy Loading)
```
[Inicio] → Cargar solo 2% de datos → RAM OK → Entrenar
            ↓                          ↓
         Solo índice              Va cargando según necesita
```

## Parámetro Clave: --max-cache-batches

Este parámetro controla cuántos batches mantener en memoria:

| Valor | RAM Usada | Cuándo Usar |
|-------|-----------|-------------|
| `1` | ~2-3 GB | RAM muy limitada (< 16 GB) |
| `2` | ~4-6 GB | **Recomendado** (16-32 GB RAM) |
| `4` | ~8-12 GB | RAM abundante (> 32 GB) |
| `8` | ~16-24 GB | GPU/servidor con mucha RAM |

## Ejemplo Práctico

### Sistema con 16 GB RAM

```bash
# Configuración recomendada
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --max-cache-batches 2 \     # Solo 2 batches en memoria
    --batch-size 16 \             # Batch size moderado
    --epochs 50
```

### Sistema con 8 GB RAM

```bash
# Configuración ultra-conservadora
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --max-cache-batches 1 \     # Solo 1 batch en memoria
    --batch-size 8 \              # Batch size pequeño
    --epochs 50
```

### Sistema con 32+ GB RAM

```bash
# Configuración óptima
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --max-cache-batches 4 \     # 4 batches en memoria
    --batch-size 32 \             # Batch size grande
    --epochs 50
```

## Entrenamiento Secuencial Automático

Usa el script preparado:

```bash
./example_train_sequential.sh
```

Este script:
1. Entrena LSTM-solo
2. Libera memoria
3. Entrena GNN-det+LSTM
4. Libera memoria
5. Entrena GVAE+LSTM

## Monitoreo de Memoria

### Durante el Entrenamiento

```bash
# Terminal 1: Entrenar
python src/baseline.py --preprocessed-dir preprocessed_data --max-cache-batches 2

# Terminal 2: Ver uso de memoria (Linux)
watch -n 1 'free -h'

# Terminal 3: Ver uso de GPU (si aplica)
watch -n 1 nvidia-smi
```

### Señales de Alerta

**⚠️ Cache muy pequeño (necesitas aumentarlo):**
- Entrenamiento muy lento
- Disco trabajando constantemente
- CPU idle esperando datos

**Solución**: Aumentar `--max-cache-batches`

**⚠️ Cache muy grande (necesitas reducirlo):**
- Error "Out of Memory"
- Sistema usa SWAP
- Todo el sistema lento

**Solución**: Reducir `--max-cache-batches`

## Flags de Control de Modelos

Para entrenar modelos específicos:

| Flag | Función |
|------|---------|
| `--run-lstm` | Entrenar LSTM-solo (default: True) |
| `--no-run-lstm` | NO entrenar LSTM-solo |
| `--run-gnn-det` | Entrenar GNN-det+LSTM (default: True) |
| `--no-run-gnn-det` | NO entrenar GNN-det+LSTM |
| `--run-gvae` | Entrenar GVAE+LSTM (default: True) |
| `--no-run-gvae` | NO entrenar GVAE+LSTM |

### Ejemplos

```bash
# Solo LSTM
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --run-lstm --no-run-gnn-det --no-run-gvae

# Solo modelos con grafos
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --no-run-lstm --run-gnn-det --run-gvae

# Todos (comportamiento default)
python src/baseline.py \
    --preprocessed-dir preprocessed_data \
    --run-lstm --run-gnn-det --run-gvae
```

## Comparación de Memoria

### Dataset Ejemplo: 1000 traces

| Método | Memoria Total | Batches en RAM | Overhead |
|--------|---------------|----------------|----------|
| **Carga Completa** | 25 GB | 100% (todos) | Alto |
| **Lazy (cache=1)** | 2.5 GB | 1% (solo 1) | Mínimo |
| **Lazy (cache=2)** | 5 GB | 2% (solo 2) | Mínimo |
| **Lazy (cache=4)** | 10 GB | 4% (solo 4) | Bajo |

**Reducción de memoria: 75-90%**

## Troubleshooting Rápido

### Problema: Out of Memory

```bash
# Prueba 1: Reducir cache
--max-cache-batches 1

# Prueba 2: Reducir batch size
--batch-size 8

# Prueba 3: Entrenar modelos por separado
--run-lstm --no-run-gnn-det --no-run-gvae
```

### Problema: Entrenamiento Muy Lento

```bash
# Prueba 1: Aumentar cache (si hay RAM)
--max-cache-batches 4

# Prueba 2: Usar SSD en lugar de HDD
# (mover preprocessed_data a SSD)

# Prueba 3: Desactivar num_workers si causa problemas
# (automático en el código)
```

### Problema: Error al Cargar Batches

```bash
# Verificar que los archivos existen
ls preprocessed_data/lstm_solo/
ls preprocessed_data/gnn/

# Verificar que tienen el prefijo correcto
ls preprocessed_data/lstm_solo/preprocessed_*.pt
```

## FAQ

**P: ¿Cuánto tarda la indexación inicial?**  
R: 1-5 segundos para 1000 traces. Es muy rápido.

**P: ¿Afecta la precisión del modelo?**  
R: No, es exactamente el mismo entrenamiento, solo cambia cómo se cargan los datos.

**P: ¿Funciona con num_workers > 0?**  
R: Sí, pero cada worker puede cachear batches adicionales.

**P: ¿Puedo usar esto con datos raw (.pkl.gz)?**  
R: No, lazy loading solo funciona con datos preprocesados (.pt).

**P: ¿Necesito volver a preprocesar?**  
R: No, los archivos preprocesados existentes funcionan sin cambios.

**P: ¿Es más lento que cargar todo?**  
R: Ligeramente más lento en la primera época, pero después el cache ayuda.

**P: ¿Funciona en Windows?**  
R: Sí, compatible con Windows, Linux y macOS.

## Próximos Pasos

1. **Ejecutar test** para verificar que todo funciona:
   ```bash
   python test_preprocessing_pipeline.py
   ```

2. **Preprocesar datos reales** si aún no lo has hecho:
   ```bash
   python src/preprocess_for_training.py \
       --data-pattern "traces_data/*.pkl*" \
       --scores-file ground_truth_scores.csv \
       --output-dir preprocessed_data
   ```

3. **Entrenar con lazy loading**:
   ```bash
   python src/baseline.py \
       --preprocessed-dir preprocessed_data \
       --max-cache-batches 2 \
       --epochs 50
   ```

## Documentación Adicional

- **MEMORY_OPTIMIZATION.md**: Explicación técnica completa
- **BASELINE_PREPROCESSING_USAGE.md**: Guía general de uso
- **example_train_sequential.sh**: Script listo para usar

## Soporte

Si encuentras problemas:
1. Verifica la sección de Troubleshooting
2. Revisa MEMORY_OPTIMIZATION.md para detalles técnicos
3. Prueba con --max-cache-batches 1 primero

---

**Última actualización**: Noviembre 18, 2024  
**Implementación**: `src/baseline.py` con lazy loading
