#!/usr/bin/env python3
"""
Script para visualizar los resultados de entrenamiento del modelo DynGAD.
Genera gráficas de loss y AUROC de validación.

Uso:
    python src/visualize_dyngad_results.py

El script busca automáticamente el archivo de resultados más reciente en:
    - dyngad_results/GraphSequenceClassifier_results_*.json

Y genera las gráficas en el directorio visualizations/ con dos tipos de archivos:
    - dyngad_losses_latest.png
    - dyngad_auroc_latest.png
"""

import json
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def find_latest_result_file(pattern):
    """Encuentra el archivo más reciente que coincida con el patrón."""
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_result_file(filepath):
    """Carga un archivo JSON de resultados."""
    if not filepath or not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error al cargar {filepath}: {e}")
        return None

def parse_dyngad_history(history_dict):
    """
    Convierte el historial de DynGAD (dict de dicts) en un dict de listas.
    """
    if not isinstance(history_dict, dict):
        return None

    train_task_losses = []
    train_aux_losses = []
    val_task_losses = []
    val_aux_losses = []
    val_aurocs = []
    
    epochs = sorted(history_dict.keys(), key=lambda x: int(x))

    for epoch in epochs:
        epoch_data = history_dict[epoch]
        train_task_losses.append(epoch_data.get('train_task_loss', np.nan))
        train_aux_losses.append(epoch_data.get('train_aux_loss', np.nan))
        val_task_losses.append(epoch_data.get('val_task_loss', np.nan))
        val_aux_losses.append(epoch_data.get('val_aux_loss', np.nan))
        
        if 'val_metrics' in epoch_data and 'auroc' in epoch_data['val_metrics']:
            val_aurocs.append(epoch_data['val_metrics']['auroc'])
            
    return {
        'train_task_loss': train_task_losses,
        'train_aux_loss': train_aux_losses,
        'val_task_loss': val_task_losses,
        'val_aux_loss': val_aux_losses,
        'val_auroc': val_aurocs
    }

def plot_losses(history, output_dir):
    """
    Genera una grilla de 2x2 de losses para el modelo DynGAD.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    color = '#2ca02c'
    label = 'DynGAD (GINE-VAE+LSTM)'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle('Training & Validation Losses', fontsize=16, fontweight='bold')
    
    train_task_loss = history.get('train_task_loss', [])
    val_task_loss = history.get('val_task_loss', [])
    train_aux_loss = history.get('train_aux_loss', [])
    val_aux_loss = history.get('val_aux_loss', [])
    
    if not train_task_loss:
        print("No se encontraron datos de loss para graficar.")
        plt.close(fig)
        return
        
    epochs = range(1, len(train_task_loss) + 1)
    
    # --- Fila 1: Task Loss ---
    axes[0, 0].plot(epochs, train_task_loss, label='Train', color=color, linewidth=2, marker='o', markersize=3, alpha=0.8)
    axes[0, 0].set_title('Training')
    axes[0, 0].set_ylabel('Task Loss', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, val_task_loss, label='Validation', color=color, linewidth=2, marker='o', markersize=3)
    axes[0, 1].set_title('Validation')
    axes[0, 1].grid(True, alpha=0.3)
    
    # --- Fila 2: Aux Loss ---
    axes[1, 0].plot(epochs, train_aux_loss, label='Train', color='orange', linewidth=2, marker='o', markersize=3, alpha=0.8)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Aux Loss', fontsize=12)
    axes[1, 0].set_yscale('log') # Escala logarítmica para Aux Loss
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, val_aux_loss, label='Validation', color='orange', linewidth=2, marker='o', markersize=3)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_yscale('log') # Escala logarítmica para Aux Loss
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'dyngad_losses_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica de Loss guardada en: {output_path}")
    
    output_path_latest = os.path.join(output_dir, 'dyngad_losses_latest.png')
    plt.savefig(output_path_latest, dpi=300, bbox_inches='tight')
    print(f"Gráfica de Loss guardada en: {output_path_latest}")
    
    plt.close(fig)

def plot_auroc(history, output_dir):
    """
    Genera gráfica de AUROC de validación para el modelo DynGAD.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    color = '#2ca02c'
    label = 'DynGAD (GINE-VAE+LSTM)'
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    val_auroc = history.get('val_auroc', [])
    
    if not val_auroc:
        print("No se encontraron datos de AUROC de validación para graficar.")
        plt.close(fig)
        return
        
    epochs = range(1, len(val_auroc) + 1)
    
    ax.plot(epochs, val_auroc, label=label, color=color, linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Validation AUROC', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'dyngad_auroc_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica de AUROC guardada en: {output_path}")
    
    output_path_latest = os.path.join(output_dir, 'dyngad_auroc_latest.png')
    plt.savefig(output_path_latest, dpi=300, bbox_inches='tight')
    print(f"Gráfica de AUROC guardada en: {output_path_latest}")
    
    plt.close(fig)

def main():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "dyngad_results"
    visualizations_dir = project_root / "visualizations"
    
    print("Buscando archivo de resultados de DynGAD más reciente...")
    print(f"Directorio de resultados: {results_dir}")
    
    pattern = str(results_dir / "GraphSequenceClassifier_results_*.json")
    latest_file = find_latest_result_file(pattern)
    
    if not latest_file:
        print(f"No se encontró ningún archivo de resultados con el patrón: {pattern}")
        return

    print(f"Encontrado: {latest_file}")
    data = load_result_file(latest_file)
    
    if not data:
        return

    history_parsed = parse_dyngad_history(data)
    if not history_parsed:
        print("No se pudo parsear el historial del archivo de resultados.")
        return

    print("\nGenerando gráficas de loss...")
    plot_losses(history_parsed, str(visualizations_dir))
    
    print("\nGenerando gráficas de AUROC...")
    plot_auroc(history_parsed, str(visualizations_dir))

if __name__ == "__main__":
    main()
