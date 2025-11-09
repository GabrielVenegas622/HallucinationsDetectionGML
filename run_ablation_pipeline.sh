#!/bin/bash
# Script de ejemplo para ejecutar experimentos de ablación completos

set -e  # Exit on error

echo "=================================="
echo "PIPELINE COMPLETO DE ABLACIÓN"
echo "=================================="

# Configuración
MODEL="llama2_chat_7B"
DATASET="triviaqa"
NUM_SAMPLES=5000
TRACES_DIR="./traces_data"
OUTPUT_DIR="./ablation_results"

# Paso 1: Extraer traces (comentar si ya están extraídos)
echo ""
echo "Paso 1/3: Extrayendo traces del modelo..."
echo "-----------------------------------"
# python src/trace_extractor.py \
#     --model $MODEL \
#     --dataset $DATASET \
#     --num-samples $NUM_SAMPLES

# Paso 2: Generar ground truth scores con BLEURT
echo ""
echo "Paso 2/3: Generando ground truth con BLEURT..."
echo "-----------------------------------"
python src/trace_to_gt.py \
    --dataset $DATASET \
    --traces-dir $TRACES_DIR \
    --output ground_truth_${DATASET}.csv

# Paso 3: Ejecutar experimentos de ablación
echo ""
echo "Paso 3/3: Ejecutando experimentos de ablación..."
echo "-----------------------------------"
python src/baseline.py \
    --data-pattern "${TRACES_DIR}/*${DATASET}*.pkl" \
    --scores-file "ground_truth_${DATASET}.csv" \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001 \
    --gnn-hidden 128 \
    --latent-dim 64 \
    --lstm-hidden 256 \
    --num-lstm-layers 2 \
    --dropout 0.3 \
    --kl-weight 0.001 \
    --output-dir $OUTPUT_DIR \
    --run-lstm \
    --run-gnn-det \
    --run-gvae

echo ""
echo "=================================="
echo "✅ PIPELINE COMPLETADO"
echo "=================================="
echo ""
echo "Resultados disponibles en:"
echo "  - ${OUTPUT_DIR}/ablation_results_*.json"
echo "  - best_lstm_baseline.pt"
echo "  - best_gnn_det_lstm.pt"
echo "  - best_gvae_lstm.pt"
echo ""
echo "Para analizar resultados:"
echo "  cat ${OUTPUT_DIR}/ablation_results_*.json | python -m json.tool"
