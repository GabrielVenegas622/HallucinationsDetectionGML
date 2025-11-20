#!/usr/bin/env python3
"""
Script para visualizar los resultados de entrenamiento de los modelos baseline.
Genera gráficas de loss, AUROC de entrenamiento y validación para cada modelo.

Uso:
    python visualize_baseline.py

El script busca automáticamente los archivos de resultados más recientes en:
    - ablation_results/lstm_only_*.json
    - ablation_results/gnn_det_lstm_*.json
    - ablation_results/gnn_vae_lstm_*.json

Y genera las gráficas en el directorio visualizations/ con dos tipos de archivos:
    - baseline_losses_<timestamp>.png y baseline_losses_latest.png
    - baseline_auroc_<timestamp>.png y baseline_auroc_latest.png

El script funciona incluso si faltan algunos de los archivos de resultados,
generando gráficas solo con los modelos disponibles.
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
    # Ordenar por tiempo de modificación (más reciente primero)
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

def extract_history(data):
    """Extrae el historial de entrenamiento del archivo de resultados."""
    if not data:
        return None
    
    # Buscar el history en diferentes ubicaciones posibles
    # Estructura nueva: data['metrics']['history']
    if 'metrics' in data and 'history' in data['metrics']:
        return data['metrics']['history']
    # Estructura alternativa: data['history']
    elif 'history' in data:
        return data['history']
    elif 'training_history' in data:
        return data['training_history']
    
    return None

def plot_losses(results_data, output_dir):
    """
    Genera gráficas de loss para los modelos disponibles.
    
    Args:
        results_data: dict con keys 'lstm', 'gnn_det', 'gnn_vae' y valores que son los datos cargados
        output_dir: directorio donde guardar las gráficas
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Colores para cada modelo
    colors = {
        'lstm': '#1f77b4',
        'gnn_det': '#ff7f0e',
        'gnn_vae': '#2ca02c'
    }
    
    labels = {
        'lstm': 'LSTM-solo',
        'gnn_det': 'GNN-Det+LSTM',
        'gnn_vae': 'GNN-VAE+LSTM'
    }
    
    # Crear figura con 2 subplots (train loss y val loss)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    has_data = False
    
    for model_key, data in results_data.items():
        if data is None:
            print(f"No hay datos disponibles para {labels[model_key]}")
            continue
        
        history = extract_history(data)
        if history is None:
            print(f"No se encontró historial de entrenamiento para {labels[model_key]}")
            continue
        
        # Extraer losses
        train_losses = history.get('train_loss', [])
        val_losses = history.get('val_loss', [])
        
        if not train_losses and not val_losses:
            print(f"No se encontraron losses para {labels[model_key]}")
            continue
        
        has_data = True
        epochs = range(1, len(train_losses) + 1)
        
        # Plot train loss
        if train_losses:
            axes[0].plot(epochs, train_losses, 
                        label=labels[model_key], 
                        color=colors[model_key],
                        linewidth=2,
                        marker='o',
                        markersize=3)
        
        # Plot val loss
        if val_losses:
            axes[1].plot(epochs, val_losses, 
                        label=labels[model_key], 
                        color=colors[model_key],
                        linewidth=2,
                        marker='o',
                        markersize=3)
    
    if not has_data:
        print("No se encontraron datos para generar las gráficas")
        plt.close(fig)
        return
    
    # Configurar subplot de train loss
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Configurar subplot de val loss
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'baseline_losses_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada en: {output_path}")
    
    # También guardar en un archivo sin timestamp para fácil acceso
    output_path_latest = os.path.join(output_dir, 'baseline_losses_latest.png')
    plt.savefig(output_path_latest, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada en: {output_path_latest}")
    
    plt.close(fig)

def plot_auroc(results_data, output_dir):
    """
    Genera gráficas de AUROC para los modelos disponibles.
    
    Args:
        results_data: dict con keys 'lstm', 'gnn_det', 'gnn_vae' y valores que son los datos cargados
        output_dir: directorio donde guardar las gráficas
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Colores para cada modelo
    colors = {
        'lstm': '#1f77b4',
        'gnn_det': '#ff7f0e',
        'gnn_vae': '#2ca02c'
    }
    
    labels = {
        'lstm': 'LSTM-solo',
        'gnn_det': 'GNN-Det+LSTM',
        'gnn_vae': 'GNN-VAE+LSTM'
    }
    
    # Crear figura con 2 subplots (train AUROC y val AUROC)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    has_data = False
    
    for model_key, data in results_data.items():
        if data is None:
            continue
        
        history = extract_history(data)
        if history is None:
            continue
        
        # Extraer AUROC (puede estar como 'train_auroc', 'train_auc', etc.)
        train_auroc = history.get('train_auroc', history.get('train_auc', history.get('train_roc_auc', [])))
        val_auroc = history.get('val_auroc', history.get('val_auc', history.get('val_roc_auc', [])))
        
        if not train_auroc and not val_auroc:
            print(f"No se encontraron datos de AUROC para {labels[model_key]}")
            continue
        
        has_data = True
        epochs = range(1, len(train_auroc) + 1) if train_auroc else range(1, len(val_auroc) + 1)
        
        # Plot train AUROC
        if train_auroc:
            axes[0].plot(epochs, train_auroc, 
                        label=labels[model_key], 
                        color=colors[model_key],
                        linewidth=2,
                        marker='o',
                        markersize=3)
        
        # Plot val AUROC
        if val_auroc:
            axes[1].plot(epochs, val_auroc, 
                        label=labels[model_key], 
                        color=colors[model_key],
                        linewidth=2,
                        marker='o',
                        markersize=3)
    
    if not has_data:
        print("No se encontraron datos de AUROC para generar las gráficas")
        plt.close(fig)
        return
    
    # Configurar subplot de train AUROC
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('AUROC', fontsize=12)
    axes[0].set_title('Training AUROC', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.5, 1.0])  # AUROC va de 0.5 a 1.0
    
    # Configurar subplot de val AUROC
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('AUROC', fontsize=12)
    axes[1].set_title('Validation AUROC', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.5, 1.0])  # AUROC va de 0.5 a 1.0
    
    plt.tight_layout()
    
    # Guardar figura
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'baseline_auroc_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada en: {output_path}")
    
    # También guardar en un archivo sin timestamp para fácil acceso
    output_path_latest = os.path.join(output_dir, 'baseline_auroc_latest.png')
    plt.savefig(output_path_latest, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada en: {output_path_latest}")
    
    plt.close(fig)

def main():
    # Directorio base del proyecto
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "ablation_results"
    visualizations_dir = project_root / "visualizations"
    
    print("Buscando archivos de resultados más recientes...")
    print(f"Directorio de resultados: {results_dir}")
    
    # Buscar archivos más recientes para cada modelo
    patterns = {
        'lstm': str(results_dir / "partial_lstm_solo_*.json"),
        'gnn_det': str(results_dir / "partial_gnn_det_lstm_*.json"),
        'gnn_vae': str(results_dir / "partial_gnn_vae_lstm_*.json")
    }
    
    results_data = {}
    
    for model_key, pattern in patterns.items():
        print(f"\nBuscando: {pattern}")
        latest_file = find_latest_result_file(pattern)
        
        if latest_file:
            print(f"Encontrado: {latest_file}")
            results_data[model_key] = load_result_file(latest_file)
        else:
            print(f"No se encontró archivo para patrón: {pattern}")
            results_data[model_key] = None
    
    # Generar gráficas de loss
    print("\nGenerando gráficas de loss...")
    plot_losses(results_data, str(visualizations_dir))
    
    # Generar gráficas de AUROC
    print("\nGenerando gráficas de AUROC...")
    plot_auroc(results_data, str(visualizations_dir))

if __name__ == "__main__":
    main()
