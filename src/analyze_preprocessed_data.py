"""
Script para analizar los archivos .pt generados por 'preprocess_for_training.py'.

Este script lee los datos de los grafos (específicamente del subdirectorio 'gnn'), 
recopila todos los valores de atención de los arcos ('edge_attr') y calcula
estadísticas descriptivas para entender su distribución.

Características:
- Itera sobre múltiples archivos .pt usando un patrón glob.
- Extrae los tensores 'edge_attr' de cada grafo en cada batch.
- Calcula estadísticas como mínimo, máximo, media, mediana y quartiles.
- Muestra un histograma de texto para visualizar la distribución de los valores.
- Diseñado para ser ejecutado desde la línea de comandos.

Dependencias:
- torch
- numpy
- tqdm
- torch_geometric (necesario para cargar los objetos `Data` guardados)

Uso:
    python src/analyze_preprocessed_data.py --data-pattern "preprocessed_data/gnn/*.pt"
"""

import torch
import argparse
import glob
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Es necesario importar Data para que torch.load pueda deserializar los objetos correctamente.
from torch_geometric.data import Data


def analyze_attention_files(data_pattern, num_bins=20):
    """
    Analiza los archivos .pt preprocesados para extraer estadísticas 
    de los valores de atención en los arcos de los grafos.
    """
    print("=" * 80)
    print("Análisis de Atenciones en Grafos Pre-procesados")
    print("=" * 80)
    
    file_paths = sorted(glob.glob(data_pattern))
    
    if not file_paths:
        print(f"⚠️ No se encontraron archivos con el patrón: {data_pattern}")
        return

    print(f"🔍 Encontrados {len(file_paths)} archivos para analizar.")

    all_attention_values = []
    total_graphs_processed = 0
    total_edges_found = 0

    for file_path in tqdm(file_paths, desc="Procesando archivos"):
        try:
            # Cargar en la CPU para evitar uso de VRAM
            data = torch.load(file_path, map_location=torch.device('cpu'))
            
            # La estructura es {'graphs': [trace_1, trace_2, ...], ...}
            # donde trace_i es [graph_layer_0, graph_layer_1, ...]
            graphs_per_batch = data.get('graphs')
            
            if not graphs_per_batch:
                tqdm.write(f"  -> Archivo {Path(file_path).name} no contiene la clave 'graphs'. Saltando.")
                continue

            for trace_graphs in graphs_per_batch:
                for graph in trace_graphs:
                    total_graphs_processed += 1
                    if graph.edge_attr is not None and graph.edge_attr.numel() > 0:
                        # edge_attr es float16, lo mantenemos así por ahora
                        all_attention_values.append(graph.edge_attr)
                        total_edges_found += graph.edge_attr.numel()

        except Exception as e:
            tqdm.write(f"\n⚠️ Error procesando el archivo {Path(file_path).name}: {e}")
            continue
            
    if not all_attention_values:
        print("\n❌ No se encontraron valores de atención en los archivos procesados.")
        return

    print("\n📊 Calculando estadísticas...")
    
    # Concatenar todos los tensores en uno solo y convertir a float32 para precisión
    attentions_tensor = torch.cat(all_attention_values).float()
    
    # --- Estadísticas Descriptivas ---
    print("\n--- Estadísticas Descriptivas de los Pesos de Atención ---")
    print(f"  - Total de grafos analizados: {total_graphs_processed:,}")
    print(f"  - Total de arcos (edges) con atención: {total_edges_found:,}")
    
    min_val = attentions_tensor.min().item()
    max_val = attentions_tensor.max().item()
    mean_val = attentions_tensor.mean().item()
    std_val = attentions_tensor.std().item()
    median_val = attentions_tensor.median().item()
    
    q25, q75 = torch.quantile(attentions_tensor, torch.tensor([0.25, 0.75])).tolist()

    print(f"\n  Mínimo:   {min_val:.6f}")
    print(f"  Máximo:   {max_val:.6f}")
    print(f"  Media:    {mean_val:.6f}")
    print(f"  Mediana:  {median_val:.6f}")
    print(f"  Desv. Estándar: {std_val:.6f}")
    print(f"  Quartil 25%: {q25:.6f}")
    print(f"  Quartil 75%: {q75:.6f}")

    # --- Histograma de Distribución ---
    print("\n--- Distribución de los Pesos de Atención (Histograma) ---")
    
    try:
        counts, bins = np.histogram(attentions_tensor.numpy(), bins=num_bins)
        
        # Normalizar cuentas para que la barra más larga tenga un ancho fijo
        max_count = counts.max()
        bar_width = 50
        
        if max_count > 0:
            normalized_counts = (counts / max_count * bar_width).astype(int)
        else:
            normalized_counts = [0] * num_bins

        for i in range(num_bins):
            bin_start = bins[i]
            bin_end = bins[i+1]
            bar = '█' * normalized_counts[i]
            count = counts[i]
            
            # Formatear para alineación
            range_str = f"[{bin_start:6.4f}, {bin_end:6.4f})".ljust(20)
            count_str = f"{count:,}".rjust(12)
            
            print(f"  {range_str} | {bar} ({count_str})")
            
    except Exception as e:
        print(f"\n  No se pudo generar el histograma: {e}")

    print("\n" + "="*80)
    print("Análisis completado.")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analiza archivos .pt pre-procesados y muestra estadísticas de las atenciones.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--data-pattern',
        type=str,
        required=True,
        help='Patrón glob para los archivos .pt a analizar.\n'
             'Ejemplo: "preprocessed_data/gnn/*.pt"'
    )
    
    parser.add_argument(
        '--num-bins',
        type=int,
        default=20,
        help='Número de bins para el histograma de distribución (default: 20).'
    )
    
    args = parser.parse_args()
    
    analyze_attention_files(args.data_pattern, args.num_bins)
