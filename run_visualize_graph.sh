#!/bin/bash
# Script para ejecutar visualize_attention_graph.py con el environment "networks"

# Activar el environment de conda
eval "$(/home/gaara/mnt/miniconda3/bin/conda shell.bash hook)"
conda activate networks

# Ejecutar el script de Python con todos los argumentos pasados
python src/visualize_attention_graph.py "$@"
