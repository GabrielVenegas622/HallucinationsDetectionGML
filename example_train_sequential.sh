#!/bin/bash
# Script de ejemplo para entrenar modelos secuencialmente
# Optimizado para sistemas con memoria RAM limitada

# Configuración
PREPROCESSED_DIR="preprocessed_data"
EPOCHS=50
BATCH_SIZE=16
LR=0.001
MAX_CACHE_BATCHES=2

echo "================================================================================"
echo "ENTRENAMIENTO SECUENCIAL DE MODELOS - Optimizado para Memoria Limitada"
echo "================================================================================"
echo ""
echo "Configuración:"
echo "  - Directorio de datos: $PREPROCESSED_DIR"
echo "  - Épocas: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LR"
echo "  - Cache de batches: $MAX_CACHE_BATCHES"
echo ""

# Paso 1: Entrenar LSTM-solo
echo "================================================================================"
echo "PASO 1/3: Entrenando LSTM-solo (usa menos memoria)"
echo "================================================================================"
echo ""

python src/baseline.py \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --max-cache-batches $MAX_CACHE_BATCHES \
    --run-lstm \
    --no-run-gnn-det \
    --no-run-gvae \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --output-dir ./ablation_results/lstm_solo

if [ $? -eq 0 ]; then
    echo "✓ LSTM-solo entrenado exitosamente"
else
    echo "✗ Error entrenando LSTM-solo"
    exit 1
fi

echo ""
echo "Liberando memoria antes del siguiente modelo..."
sleep 5

# Paso 2: Entrenar GNN-det+LSTM
echo "================================================================================"
echo "PASO 2/3: Entrenando GNN-det+LSTM"
echo "================================================================================"
echo ""

python src/baseline.py \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --max-cache-batches $MAX_CACHE_BATCHES \
    --no-run-lstm \
    --run-gnn-det \
    --no-run-gvae \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --output-dir ./ablation_results/gnn_det

if [ $? -eq 0 ]; then
    echo "✓ GNN-det+LSTM entrenado exitosamente"
else
    echo "✗ Error entrenando GNN-det+LSTM"
    exit 1
fi

echo ""
echo "Liberando memoria antes del siguiente modelo..."
sleep 5

# Paso 3: Entrenar GVAE+LSTM
echo "================================================================================"
echo "PASO 3/3: Entrenando GVAE+LSTM"
echo "================================================================================"
echo ""

python src/baseline.py \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --max-cache-batches $MAX_CACHE_BATCHES \
    --no-run-lstm \
    --no-run-gnn-det \
    --run-gvae \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --kl-weight 0.001 \
    --output-dir ./ablation_results/gvae

if [ $? -eq 0 ]; then
    echo "✓ GVAE+LSTM entrenado exitosamente"
else
    echo "✗ Error entrenando GVAE+LSTM"
    exit 1
fi

# Resumen final
echo ""
echo "================================================================================"
echo "ENTRENAMIENTO COMPLETADO"
echo "================================================================================"
echo ""
echo "Resultados guardados en:"
echo "  - LSTM-solo:     ./ablation_results/lstm_solo/"
echo "  - GNN-det+LSTM:  ./ablation_results/gnn_det/"
echo "  - GVAE+LSTM:     ./ablation_results/gvae/"
echo ""
echo "Para comparar resultados, revisar los archivos ablation_results_*.json"
echo "en cada directorio."
echo ""
echo "================================================================================"
