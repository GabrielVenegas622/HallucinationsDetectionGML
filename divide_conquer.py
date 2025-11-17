#!/usr/bin/env python3
"""
Divide and Conquer: Subdivide large trace batches into smaller ones.

Este script divide archivos de batches grandes (1000 traces, ~14GB) en batches
más pequeños (250 traces, ~3.5GB) para mejorar la gestión de memoria.

Características:
- Divide batches de 1000 traces en 4 batches de 250 traces
- Soporta archivos .pkl y .pkl.gz
- Mantiene compresión gzip en archivos de salida
- Progreso visual con tqdm
- Limpieza de memoria entre operaciones
- Validación de integridad de datos

Uso:
    python divide_conquer.py --input traces_data/ --output traces_data_split/ --traces-per-batch 250
    
    # O con archivos específicos
    python divide_conquer.py --input traces_data/batch_0000.pkl.gz --output traces_data_split/
"""

import argparse
import pickle
import gzip
import glob
import os
import gc
from pathlib import Path
from tqdm import tqdm
import sys


def get_file_size_mb(file_path):
    """Obtiene el tamaño del archivo en MB."""
    return os.path.getsize(file_path) / (1024 * 1024)


def load_batch_file(file_path, verbose=True):
    """
    Carga un archivo de batch (.pkl o .pkl.gz).
    
    Args:
        file_path: Ruta al archivo
        verbose: Mostrar información
    
    Returns:
        Lista de traces
    """
    if verbose:
        size_mb = get_file_size_mb(file_path)
        print(f"\n📂 Cargando: {file_path}")
        print(f"   Tamaño: {size_mb:.2f} MB")
    
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        
        if verbose:
            print(f"   ✅ Cargado: {len(data)} traces")
        
        return data
    
    except Exception as e:
        print(f"   ❌ ERROR al cargar {file_path}: {e}")
        return None


def save_batch_file(data, file_path, compress=True, verbose=True):
    """
    Guarda un batch de traces a archivo.
    
    Args:
        data: Lista de traces
        file_path: Ruta de salida
        compress: Si True, comprime con gzip
        verbose: Mostrar información
    """
    if verbose:
        print(f"   💾 Guardando: {file_path} ({len(data)} traces)")
    
    try:
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if compress and not file_path.endswith('.gz'):
            file_path += '.gz'
        
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if verbose:
            size_mb = get_file_size_mb(file_path)
            print(f"   ✅ Guardado: {size_mb:.2f} MB")
        
        return file_path
    
    except Exception as e:
        print(f"   ❌ ERROR al guardar {file_path}: {e}")
        return None


def divide_batch(input_file, output_dir, traces_per_batch=250, compress=True, 
                 keep_original=True, verbose=True):
    """
    Divide un archivo de batch grande en varios archivos más pequeños.
    
    Args:
        input_file: Ruta al archivo de entrada
        output_dir: Directorio de salida
        traces_per_batch: Número de traces por batch de salida
        compress: Comprimir archivos de salida con gzip
        keep_original: Mantener archivo original
        verbose: Mostrar información detallada
    
    Returns:
        Lista de archivos creados
    """
    # Cargar el batch original
    data = load_batch_file(input_file, verbose=verbose)
    
    if data is None:
        return []
    
    total_traces = len(data)
    num_batches = (total_traces + traces_per_batch - 1) // traces_per_batch
    
    if verbose:
        print(f"\n📊 Dividiendo en {num_batches} batches de ~{traces_per_batch} traces")
    
    # Extraer nombre base del archivo
    input_path = Path(input_file)
    base_name = input_path.stem
    if base_name.endswith('.pkl'):
        base_name = base_name[:-4]
    
    created_files = []
    
    # Dividir en batches más pequeños
    for i in tqdm(range(num_batches), desc="Dividiendo batches", disable=not verbose):
        start_idx = i * traces_per_batch
        end_idx = min((i + 1) * traces_per_batch, total_traces)
        
        sub_batch = data[start_idx:end_idx]
        
        # Generar nombre de archivo de salida
        output_name = f"{base_name}_sub{i:04d}.pkl"
        output_path = os.path.join(output_dir, output_name)
        
        # Guardar sub-batch
        saved_path = save_batch_file(sub_batch, output_path, compress=compress, 
                                     verbose=False)
        
        if saved_path:
            created_files.append(saved_path)
        
        # Liberar memoria
        del sub_batch
        gc.collect()
    
    # Liberar memoria del batch original
    del data
    gc.collect()
    
    if verbose:
        print(f"\n✅ Creados {len(created_files)} archivos")
        total_size = sum(get_file_size_mb(f) for f in created_files)
        print(f"   Tamaño total: {total_size:.2f} MB")
    
    # Opcional: Eliminar archivo original
    if not keep_original:
        if verbose:
            print(f"\n🗑️  Eliminando archivo original: {input_file}")
        try:
            os.remove(input_file)
        except Exception as e:
            print(f"   ⚠️  No se pudo eliminar: {e}")
    
    return created_files


def validate_split(original_file, split_files, verbose=True):
    """
    Valida que la división fue correcta.
    
    Args:
        original_file: Archivo original
        split_files: Lista de archivos divididos
        verbose: Mostrar información
    
    Returns:
        True si la validación es exitosa
    """
    if verbose:
        print("\n🔍 Validando división...")
    
    # Cargar original
    original_data = load_batch_file(original_file, verbose=False)
    if original_data is None:
        return False
    
    original_count = len(original_data)
    
    # Contar traces en archivos divididos
    split_count = 0
    for split_file in tqdm(split_files, desc="Validando", disable=not verbose):
        split_data = load_batch_file(split_file, verbose=False)
        if split_data is not None:
            split_count += len(split_data)
        del split_data
        gc.collect()
    
    del original_data
    gc.collect()
    
    if verbose:
        print(f"   Original: {original_count} traces")
        print(f"   Divididos: {split_count} traces")
    
    if original_count == split_count:
        if verbose:
            print("   ✅ Validación exitosa")
        return True
    else:
        if verbose:
            print(f"   ❌ ERROR: Diferencia de {abs(original_count - split_count)} traces")
        return False


def process_directory(input_dir, output_dir, traces_per_batch=250, 
                      compress=True, keep_original=True, validate=True, 
                      pattern="*.pkl*"):
    """
    Procesa todos los archivos en un directorio.
    
    Args:
        input_dir: Directorio de entrada
        output_dir: Directorio de salida
        traces_per_batch: Número de traces por batch
        compress: Comprimir archivos de salida
        keep_original: Mantener archivos originales
        validate: Validar cada división
        pattern: Patrón de archivos a procesar
    """
    # Buscar archivos
    input_path = Path(input_dir)
    file_pattern = str(input_path / pattern)
    files = glob.glob(file_pattern)
    
    # Filtrar archivos .part
    files = [f for f in files if not f.endswith('.part')]
    
    if not files:
        print(f"❌ No se encontraron archivos en {input_dir} con patrón {pattern}")
        return
    
    print(f"📁 Encontrados {len(files)} archivos para procesar")
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    # Procesar cada archivo
    for file_path in files:
        print(f"\n{'='*80}")
        print(f"Procesando: {os.path.basename(file_path)}")
        print(f"{'='*80}")
        
        try:
            # Dividir el batch
            created_files = divide_batch(
                file_path, 
                output_dir, 
                traces_per_batch=traces_per_batch,
                compress=compress,
                keep_original=True,  # Siempre mantener durante validación
                verbose=True
            )
            
            if not created_files:
                print(f"❌ Falló la división de {file_path}")
                fail_count += 1
                continue
            
            # Validar si se solicita
            if validate:
                is_valid = validate_split(file_path, created_files, verbose=True)
                if not is_valid:
                    print(f"❌ Validación falló para {file_path}")
                    fail_count += 1
                    # Eliminar archivos creados si la validación falla
                    for f in created_files:
                        try:
                            os.remove(f)
                        except:
                            pass
                    continue
            
            # Si llegamos aquí, todo OK
            success_count += 1
            
            # Ahora sí, eliminar original si se solicita
            if not keep_original:
                print(f"\n🗑️  Eliminando archivo original: {file_path}")
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"   ⚠️  No se pudo eliminar: {e}")
        
        except Exception as e:
            print(f"❌ ERROR procesando {file_path}: {e}")
            fail_count += 1
        
        # Limpiar memoria
        gc.collect()
    
    # Resumen final
    print(f"\n{'='*80}")
    print(f"RESUMEN FINAL")
    print(f"{'='*80}")
    print(f"✅ Exitosos: {success_count}/{len(files)}")
    print(f"❌ Fallidos: {fail_count}/{len(files)}")
    print(f"📂 Archivos de salida en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Divide batches grandes de traces en batches más pequeños.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Dividir todos los archivos en un directorio
  python divide_conquer.py --input traces_data/ --output traces_data_split/

  # Dividir un archivo específico
  python divide_conquer.py --input traces_data/batch_0000.pkl.gz --output traces_data_split/

  # Dividir en batches de 100 traces
  python divide_conquer.py --input traces_data/ --output traces_data_split/ --traces-per-batch 100

  # Dividir sin comprimir
  python divide_conquer.py --input traces_data/ --output traces_data_split/ --no-compress

  # Dividir y eliminar originales (¡cuidado!)
  python divide_conquer.py --input traces_data/ --output traces_data_split/ --delete-original

  # Dividir sin validar (más rápido pero menos seguro)
  python divide_conquer.py --input traces_data/ --output traces_data_split/ --no-validate
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Archivo o directorio de entrada')
    parser.add_argument('--output', '-o', required=True,
                       help='Directorio de salida')
    parser.add_argument('--traces-per-batch', '-n', type=int, default=250,
                       help='Número de traces por batch de salida (default: 250)')
    parser.add_argument('--no-compress', action='store_true',
                       help='No comprimir archivos de salida con gzip')
    parser.add_argument('--delete-original', action='store_true',
                       help='Eliminar archivos originales después de dividir (¡usar con cuidado!)')
    parser.add_argument('--no-validate', action='store_true',
                       help='No validar la división (más rápido pero menos seguro)')
    parser.add_argument('--pattern', type=str, default='*.pkl*',
                       help='Patrón de archivos a procesar (default: *.pkl*)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Modo silencioso (menos output)')
    
    args = parser.parse_args()
    
    # Verificar que input existe
    if not os.path.exists(args.input):
        print(f"❌ ERROR: No existe {args.input}")
        sys.exit(1)
    
    # Advertencia si se va a eliminar originales
    if args.delete_original:
        print("\n⚠️  ADVERTENCIA: Los archivos originales serán ELIMINADOS")
        print("   Solo se eliminarán si la validación es exitosa.")
        response = input("   ¿Continuar? (yes/no): ")
        if response.lower() not in ['yes', 'y', 'si', 's']:
            print("Operación cancelada.")
            sys.exit(0)
    
    compress = not args.no_compress
    keep_original = not args.delete_original
    validate = not args.no_validate
    
    print("\n" + "="*80)
    print("DIVIDE AND CONQUER - División de Batches")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Traces por batch: {args.traces_per_batch}")
    print(f"Comprimir: {compress}")
    print(f"Mantener originales: {keep_original}")
    print(f"Validar: {validate}")
    print("="*80 + "\n")
    
    # Determinar si es archivo o directorio
    if os.path.isfile(args.input):
        # Procesar un solo archivo
        created_files = divide_batch(
            args.input,
            args.output,
            traces_per_batch=args.traces_per_batch,
            compress=compress,
            keep_original=True,  # Siempre mantener durante validación
            verbose=not args.quiet
        )
        
        if created_files and validate:
            is_valid = validate_split(args.input, created_files, verbose=not args.quiet)
            if not is_valid:
                print("❌ Validación falló")
                sys.exit(1)
        
        # Eliminar original si se solicita
        if not keep_original and created_files:
            print(f"\n🗑️  Eliminando archivo original: {args.input}")
            try:
                os.remove(args.input)
            except Exception as e:
                print(f"   ⚠️  No se pudo eliminar: {e}")
    
    elif os.path.isdir(args.input):
        # Procesar directorio
        process_directory(
            args.input,
            args.output,
            traces_per_batch=args.traces_per_batch,
            compress=compress,
            keep_original=keep_original,
            validate=validate,
            pattern=args.pattern
        )
    
    else:
        print(f"❌ ERROR: {args.input} no es un archivo ni directorio válido")
        sys.exit(1)
    
    print("\n✅ Proceso completado\n")


if __name__ == '__main__':
    main()
