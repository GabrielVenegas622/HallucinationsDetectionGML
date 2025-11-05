"""
Utilidad para cargar y trabajar con múltiples archivos batch de traces.
Proporciona iteradores y funciones para acceder a los datos de manera eficiente
sin cargar todo en memoria.
"""

import pickle
from pathlib import Path
from typing import Iterator, List, Dict, Optional
import numpy as np


class TraceBatchLoader:
    """
    Cargador eficiente de batches de traces.
    Permite iterar sobre todos los traces sin cargar todo en memoria.
    """
    
    def __init__(self, traces_dir: str = "./traces_data"):
        """
        Inicializa el cargador de batches.
        
        Args:
            traces_dir: Directorio donde están los archivos batch
        """
        self.traces_dir = Path(traces_dir)
        self.batch_files = sorted(self.traces_dir.glob("trivia_qa_traces_batch_*.pkl"))
        
        if not self.batch_files:
            raise FileNotFoundError(f"No se encontraron archivos batch en {traces_dir}")
        
        print(f"✅ Encontrados {len(self.batch_files)} archivos batch")
    
    def __len__(self):
        """Retorna el número de batches disponibles."""
        return len(self.batch_files)
    
    def get_batch(self, batch_idx: int) -> List[Dict]:
        """
        Carga un batch específico.
        
        Args:
            batch_idx: Índice del batch a cargar (0-based)
            
        Returns:
            Lista de traces del batch
        """
        if batch_idx < 0 or batch_idx >= len(self.batch_files):
            raise IndexError(f"Batch index {batch_idx} fuera de rango [0, {len(self.batch_files)})")
        
        with open(self.batch_files[batch_idx], 'rb') as f:
            return pickle.load(f)
    
    def iter_batches(self) -> Iterator[List[Dict]]:
        """
        Itera sobre todos los batches.
        Útil para procesar el dataset completo sin cargar todo en memoria.
        
        Yields:
            Lista de traces de cada batch
        """
        for batch_file in self.batch_files:
            with open(batch_file, 'rb') as f:
                yield pickle.load(f)
    
    def iter_traces(self) -> Iterator[Dict]:
        """
        Itera sobre todos los traces individuales.
        
        Yields:
            Un trace a la vez
        """
        for batch in self.iter_batches():
            for trace in batch:
                yield trace
    
    def get_total_traces(self) -> int:
        """
        Cuenta el total de traces en todos los batches.
        Nota: Requiere cargar cada batch para contar.
        """
        total = 0
        for batch in self.iter_batches():
            total += len(batch)
        return total
    
    def get_trace_by_global_id(self, global_id: int) -> Optional[Dict]:
        """
        Busca un trace por su ID global.
        
        Args:
            global_id: ID global del trace en el dataset
            
        Returns:
            El trace si se encuentra, None si no
        """
        for trace in self.iter_traces():
            if trace.get('global_example_id') == global_id:
                return trace
        return None
    
    def get_batch_info(self) -> List[Dict]:
        """
        Obtiene información sobre cada batch sin cargar los datos completos.
        
        Returns:
            Lista de diccionarios con info de cada batch
        """
        info = []
        for idx, batch_file in enumerate(self.batch_files):
            size_mb = batch_file.stat().st_size / (1024 * 1024)
            
            # Cargar solo para contar
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f)
                num_traces = len(batch)
            
            info.append({
                'batch_idx': idx,
                'filename': batch_file.name,
                'size_mb': size_mb,
                'num_traces': num_traces
            })
        
        return info


def merge_batches(output_file: str = "./traces_data/all_traces_merged.pkl",
                  traces_dir: str = "./traces_data"):
    """
    Combina todos los batches en un solo archivo.
    ⚠️  ADVERTENCIA: Esto puede requerir mucha RAM.
    
    Args:
        output_file: Ruta del archivo de salida
        traces_dir: Directorio con los batches
    """
    loader = TraceBatchLoader(traces_dir)
    
    print(f"⚠️  Advertencia: Esto cargará todos los traces en memoria")
    print(f"   Batches a combinar: {len(loader)}")
    
    response = input("¿Continuar? (s/n): ")
    if response.lower() != 's':
        print("Operación cancelada")
        return
    
    all_traces = []
    
    print("\n📦 Combinando batches...")
    for batch in loader.iter_batches():
        all_traces.extend(batch)
        print(f"   • Total acumulado: {len(all_traces)} traces")
    
    print(f"\n💾 Guardando en {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(all_traces, f)
    
    output_path = Path(output_file)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Guardado: {len(all_traces)} traces, {size_mb:.2f} MB")


def extract_batch_subset(batch_numbers: List[int], 
                         output_file: str,
                         traces_dir: str = "./traces_data"):
    """
    Extrae y combina solo algunos batches específicos.
    
    Args:
        batch_numbers: Lista de números de batch a extraer
        output_file: Archivo de salida
        traces_dir: Directorio con los batches
    """
    loader = TraceBatchLoader(traces_dir)
    
    selected_traces = []
    
    for batch_num in batch_numbers:
        print(f"📦 Cargando batch {batch_num}...")
        batch = loader.get_batch(batch_num)
        selected_traces.extend(batch)
        print(f"   • Añadidos {len(batch)} traces")
    
    print(f"\n💾 Guardando {len(selected_traces)} traces en {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(selected_traces, f)
    
    output_path = Path(output_file)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Guardado: {size_mb:.2f} MB")


def main():
    """Ejemplo de uso del TraceBatchLoader."""
    
    print("="*80)
    print("EJEMPLO DE USO: TraceBatchLoader")
    print("="*80)
    
    try:
        # Inicializar el loader
        loader = TraceBatchLoader()
        
        # Información de batches
        print(f"\n📊 Información de batches:")
        batch_info = loader.get_batch_info()
        
        total_traces = 0
        total_size = 0
        
        for info in batch_info:
            print(f"\nBatch {info['batch_idx']}: {info['filename']}")
            print(f"  - Traces: {info['num_traces']}")
            print(f"  - Tamaño: {info['size_mb']:.2f} MB")
            total_traces += info['num_traces']
            total_size += info['size_mb']
        
        print(f"\n{'='*80}")
        print(f"Total: {total_traces} traces en {len(batch_info)} batches")
        print(f"Tamaño total: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
        
        # Ejemplo: Iterar sobre primeros 3 traces
        print(f"\n{'='*80}")
        print("EJEMPLO: Primeros 3 traces")
        print(f"{'='*80}")
        
        for idx, trace in enumerate(loader.iter_traces()):
            if idx >= 3:
                break
            
            print(f"\nTrace #{idx}:")
            print(f"  Global ID: {trace.get('global_example_id', 'N/A')}")
            print(f"  Batch: {trace.get('batch_number', 'N/A')}")
            print(f"  Pregunta: {trace['question'][:60]}...")
            print(f"  Respuesta: {trace['generated_answer_clean'][:60]}...")
        
        # Ejemplo: Cargar un batch específico
        print(f"\n{'='*80}")
        print("EJEMPLO: Cargar batch 0 completo")
        print(f"{'='*80}")
        
        batch_0 = loader.get_batch(0)
        print(f"\nBatch 0 contiene {len(batch_0)} traces")
        
        # Estadísticas del batch
        lengths = [len(t['tokens']) - t['prompt_length'] for t in batch_0]
        print(f"Longitud promedio de respuestas: {np.mean(lengths):.2f} tokens")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Ejecuta primero trace_extractor.py para generar los batches")


if __name__ == "__main__":
    main()
