#!/usr/bin/env python3
"""
Script para dividir archivos .pt grandes en partes más pequeñas.
Divide archivos con 250 traces en 5 sub-archivos de 50 traces cada uno.
"""

import argparse
import os
import torch
from pathlib import Path
from tqdm import tqdm
import gc


def divide_batch_file(input_path: Path, output_dir: Path, traces_per_part: int = 50):
    """
    Divide un archivo .pt con múltiples traces en partes más pequeñas.
    
    Args:
        input_path: Ruta al archivo de entrada
        output_dir: Directorio donde guardar las partes
        traces_per_part: Número de traces por sub-archivo
    """
    # Cargar el archivo
    batch_data = torch.load(input_path, weights_only=False)
    
    # Validar estructura
    required_keys = ['graphs', 'labels', 'question_ids']
    for key in required_keys:
        if key not in batch_data:
            raise ValueError(f"Archivo {input_path} no contiene la clave '{key}'")
    
    graphs = batch_data['graphs']
    labels = batch_data['labels']
    question_ids = batch_data['question_ids']
    
    # Validar que tengan la misma longitud
    num_traces = len(graphs)
    if len(labels) != num_traces or len(question_ids) != num_traces:
        raise ValueError(f"Longitudes inconsistentes en {input_path}: "
                         f"graphs={len(graphs)}, labels={len(labels)}, "
                         f"question_ids={len(question_ids)}")
    
    # Calcular número de partes
    num_parts = (num_traces + traces_per_part - 1) // traces_per_part
    
    # Extraer nombre base del archivo (sin extensión)
    base_name = input_path.stem  # Ejemplo: preprocessed_llama2_chat_7B_triviaqa_batch_0000_sub0000
    
    # Dividir y guardar
    for part_idx in range(num_parts):
        start_idx = part_idx * traces_per_part
        end_idx = min(start_idx + traces_per_part, num_traces)
        
        # Slice sincronizado
        part_data = {
            'graphs': graphs[start_idx:end_idx],
            'labels': labels[start_idx:end_idx],
            'question_ids': question_ids[start_idx:end_idx]
        }
        
        # Nombre del archivo de salida
        output_name = f"{base_name}_part{part_idx}.pt"
        output_path = output_dir / output_name
        
        # Guardar
        torch.save(part_data, output_path)
    
    # Liberar memoria
    del batch_data, graphs, labels, question_ids
    gc.collect()
    
    return num_parts


def process_directory(input_dir: Path, output_dir: Path, traces_per_part: int = 50):
    """
    Procesa todos los archivos .pt en un directorio.
    
    Args:
        input_dir: Directorio de entrada con archivos .pt
        output_dir: Directorio de salida para las partes
        traces_per_part: Número de traces por sub-archivo
    """
    # Crear directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Buscar todos los archivos .pt
    pt_files = sorted(input_dir.glob("*.pt"))
    
    if not pt_files:
        print(f"No se encontraron archivos .pt en {input_dir}")
        return
    
    print(f"Encontrados {len(pt_files)} archivos para dividir")
    print(f"Dividiendo en sub-archivos de {traces_per_part} traces cada uno")
    print(f"Directorio de salida: {output_dir}\n")
    
    total_parts = 0
    
    # Procesar cada archivo con barra de progreso
    for pt_file in tqdm(pt_files, desc="Procesando archivos"):
        try:
            num_parts = divide_batch_file(pt_file, output_dir, traces_per_part)
            total_parts += num_parts
        except Exception as e:
            print(f"\nError procesando {pt_file.name}: {e}")
            continue
    
    print(f"\n✓ Completado! Se generaron {total_parts} sub-archivos en total")


def main():
    parser = argparse.ArgumentParser(
        description="Divide archivos .pt grandes en partes más pequeñas"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help="Directorio con archivos .pt de entrada"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help="Directorio donde guardar las partes divididas"
    )
    parser.add_argument(
        '--traces-per-part',
        type=int,
        default=50,
        help="Número de traces por sub-archivo (default: 50)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validar directorio de entrada
    if not input_dir.exists():
        print(f"Error: El directorio de entrada no existe: {input_dir}")
        return
    
    if not input_dir.is_dir():
        print(f"Error: La ruta de entrada no es un directorio: {input_dir}")
        return
    
    # Procesar
    process_directory(input_dir, output_dir, args.traces_per_part)


if __name__ == "__main__":
    main()
