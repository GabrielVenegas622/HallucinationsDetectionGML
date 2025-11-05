"""
Script de ejemplo para cargar y explorar los traces extraídos.
Útil para verificar que los datos se guardaron correctamente.
"""

import pickle
import numpy as np
from pathlib import Path


def load_traces(traces_file):
    """Carga los traces desde el archivo pickle."""
    with open(traces_file, 'rb') as f:
        traces = pickle.load(f)
    return traces


def analyze_trace(trace, trace_idx=0):
    """Analiza un trace individual y muestra información detallada."""
    print(f"\n{'='*80}")
    print(f"ANÁLISIS DEL TRACE #{trace_idx}")
    print(f"{'='*80}")
    
    # Información básica
    print(f"\n📝 Pregunta: {trace['question']}")
    print(f"\n💬 Respuesta Limpia: {trace['generated_answer_clean']}")
    
    if trace.get('ground_truth_answers'):
        print(f"\n✅ Respuestas Correctas:")
        for i, ans in enumerate(trace['ground_truth_answers'][:3], 1):
            print(f"   {i}. {ans}")
    
    # Información de tokens
    num_tokens_generated = len(trace['tokens'])  # Ahora tokens solo contiene la respuesta
    print(f"\n🔢 Tokens:")
    print(f"   - Tokens en respuesta limpia: {num_tokens_generated}")
    print(f"   - Tokens antes del corte: {trace.get('tokens_before_cutoff', num_tokens_generated)}")
    print(f"   - Tokens descartados: {trace.get('tokens_after_cutoff', 0)}")
    print(f"   - Método de corte: {trace.get('cutoff_method', 'N/A')}")
    
    # Mostrar tokens decodificados si están disponibles
    if 'tokens_decoded' in trace:
        print(f"\n📝 Tokens decodificados (primeros 10):")
        for i, token_text in enumerate(trace['tokens_decoded'][:10]):
            print(f"   {i}: '{token_text}'")
    
    # Información de capas
    num_layers = trace['num_layers']
    print(f"\n🏗️  Arquitectura:")
    print(f"   - Número de capas: {num_layers}")
    
    # Análisis de hidden states
    print(f"\n🧠 Hidden States:")
    print(f"   - Estructura: {len(trace['hidden_states'])} capas")
    if trace['hidden_states']:
        # Analizar primera capa
        first_layer = trace['hidden_states'][0]
        print(f"   - Tokens capturados por capa: {len(first_layer)}")
        if first_layer:
            first_state = first_layer[0]
            print(f"   - Shape de cada estado: {first_state.shape}")
            print(f"   - Dimensión oculta: {first_state.shape[-1]}")
    
    # Análisis de atenciones
    print(f"\n👁️  Attention Matrices:")
    print(f"   - Estructura: {len(trace['attentions'])} capas")
    if trace['attentions']:
        # Analizar primera capa
        first_layer_attn = trace['attentions'][0]
        print(f"   - Tokens capturados por capa: {len(first_layer_attn)}")
        if first_layer_attn:
            first_attn = first_layer_attn[0]
            print(f"   - Shape de cada matriz: {first_attn.shape}")
            batch, num_heads, seq_len, _ = first_attn.shape
            print(f"   - Número de cabezas: {num_heads}")
            print(f"   - Secuencia máxima: {seq_len}")
            
            # Estadísticas de la primera matriz de atención
            print(f"\n   📊 Estadísticas (primera matriz, promedio sobre cabezas):")
            avg_attn = first_attn[0].mean(axis=0)  # Promediar sobre cabezas
            print(f"      - Media: {avg_attn.mean():.4f}")
            print(f"      - Std: {avg_attn.std():.4f}")
            print(f"      - Min: {avg_attn.min():.4f}")
            print(f"      - Max: {avg_attn.max():.4f}")


def analyze_dataset_statistics(traces):
    """Analiza estadísticas globales del dataset."""
    print(f"\n{'='*80}")
    print(f"ESTADÍSTICAS DEL DATASET COMPLETO")
    print(f"{'='*80}")
    
    num_traces = len(traces)
    print(f"\n📊 Tamaño del dataset: {num_traces} ejemplos")
    
    # Longitudes de respuestas
    answer_lengths = []
    cutoff_methods = {}
    
    for trace in traces:
        # Ahora tokens solo contiene la respuesta limpia
        num_generated = len(trace['tokens'])
        answer_lengths.append(num_generated)
        
        # Recopilar métodos de corte
        method = trace.get('cutoff_method', 'unknown')
        cutoff_methods[method] = cutoff_methods.get(method, 0) + 1
    
    print(f"\n📏 Longitud de respuestas limpias:")
    print(f"   - Media: {np.mean(answer_lengths):.2f} tokens")
    print(f"   - Mediana: {np.median(answer_lengths):.2f} tokens")
    print(f"   - Min: {np.min(answer_lengths)} tokens")
    print(f"   - Max: {np.max(answer_lengths)} tokens")
    print(f"   - Std: {np.std(answer_lengths):.2f} tokens")
    
    # Mostrar métodos de corte
    if cutoff_methods:
        print(f"\n✂️  Métodos de corte utilizados:")
        for method, count in sorted(cutoff_methods.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / num_traces) * 100
            print(f"   • {method}: {count} ({percentage:.1f}%)")
    
    # Verificar consistencia
    num_layers_list = [trace['num_layers'] for trace in traces]
    unique_layers = set(num_layers_list)
    print(f"\n🏗️  Capas por modelo: {unique_layers}")
    
    # Tamaño en memoria
    import sys
    size_mb = sys.getsizeof(pickle.dumps(traces)) / (1024 * 1024)
    print(f"\n💾 Tamaño estimado en memoria: {size_mb:.2f} MB")


def main():
    # Buscar archivos de traces
    traces_dir = Path("./traces_data")
    
    if not traces_dir.exists():
        print(f"❌ No se encontró el directorio {traces_dir}")
        print("   Ejecuta primero trace_extractor.py")
        return
    
    # Buscar archivos pickle (batch y archivos antiguos)
    batch_files = sorted(traces_dir.glob("trivia_qa_traces_batch_*.pkl"))
    old_files = list(traces_dir.glob("trivia_qa_traces_*.pkl"))
    old_files = [f for f in old_files if "batch" not in f.name]
    
    if not batch_files and not old_files:
        print(f"❌ No se encontraron archivos .pkl en {traces_dir}")
        return
    
    # Mostrar información sobre archivos encontrados
    print(f"{'='*80}")
    print("ARCHIVOS DE TRACES ENCONTRADOS")
    print(f"{'='*80}\n")
    
    if batch_files:
        print(f"✅ Archivos en batch: {len(batch_files)}")
        total_size = 0
        for f in batch_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"   • {f.name}: {size_mb:.2f} MB")
        print(f"\n💾 Tamaño total de batches: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    if old_files:
        print(f"\n📦 Archivos individuales (formato antiguo): {len(old_files)}")
        for f in old_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   • {f.name}: {size_mb:.2f} MB")
    
    # Cargar y analizar batches
    if batch_files:
        print(f"\n{'='*80}")
        print("ANÁLISIS DE BATCHES")
        print(f"{'='*80}\n")
        
        all_traces_count = 0
        answer_lengths = []
        cutoff_methods = {}
        
        # Analizar cada batch
        for batch_idx, batch_file in enumerate(batch_files):
            print(f"📂 Cargando batch {batch_idx}: {batch_file.name}...")
            
            try:
                with open(batch_file, 'rb') as f:
                    traces = load_traces(batch_file)
                
                num_traces = len(traces)
                all_traces_count += num_traces
                
                print(f"   ✅ {num_traces} traces en este batch")
                
                # Recopilar estadísticas
                for trace in traces:
                    num_generated = len(trace['tokens'])  # Ahora solo contiene respuesta
                    answer_lengths.append(num_generated)
                    
                    # Métodos de corte
                    method = trace.get('cutoff_method', 'unknown')
                    cutoff_methods[method] = cutoff_methods.get(method, 0) + 1
                
                # Mostrar ejemplo del primer batch
                if batch_idx == 0 and traces:
                    analyze_trace(traces[0], 0)
                
            except Exception as e:
                print(f"   ❌ Error cargando batch {batch_idx}: {e}")
        
        # Estadísticas globales
        print(f"\n{'='*80}")
        print("ESTADÍSTICAS GLOBALES DEL DATASET")
        print(f"{'='*80}\n")
        
        print(f"📊 Total de traces en todos los batches: {all_traces_count}")
        
        if answer_lengths:
            print(f"\n📏 Longitud de respuestas limpias:")
            print(f"   - Media: {np.mean(answer_lengths):.2f} tokens")
            print(f"   - Mediana: {np.median(answer_lengths):.2f} tokens")
            print(f"   - Min: {np.min(answer_lengths)} tokens")
            print(f"   - Max: {np.max(answer_lengths)} tokens")
            print(f"   - Std: {np.std(answer_lengths):.2f} tokens")
        
        # Mostrar métodos de corte
        if cutoff_methods:
            print(f"\n✂️  Métodos de corte utilizados:")
            for method, count in sorted(cutoff_methods.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / all_traces_count) * 100 if all_traces_count > 0 else 0
                print(f"   • {method}: {count} ({percentage:.1f}%)")
        
        # Mostrar algunos ejemplos de diferentes batches
        print(f"\n{'='*80}")
        print("EJEMPLOS DE DIFERENTES BATCHES")
        print(f"{'='*80}")
        
        for batch_idx in [0, len(batch_files)//2, len(batch_files)-1]:
            if batch_idx < len(batch_files):
                print(f"\n--- Ejemplo del batch {batch_idx} ---")
                with open(batch_files[batch_idx], 'rb') as f:
                    traces = pickle.load(f)
                if traces:
                    trace = traces[0]
                    num_gen = len(trace['tokens'])  # Solo respuesta
                    print(f"  Q: {trace['question'][:60]}...")
                    print(f"  A: {trace['generated_answer_clean'][:60]}...")
                    print(f"  Tokens generados: {num_gen}")
                    print(f"  Método de corte: {trace.get('cutoff_method', 'N/A')}")
                    print(f"  Batch number: {trace.get('batch_number', 'N/A')}")
                    print(f"  Global ID: {trace.get('global_example_id', 'N/A')}")
                    if 'tokens_decoded' in trace:
                        print(f"  Tokens: {trace['tokens_decoded'][:5]}...")
        
    # Si hay archivos antiguos, también analizarlos
    elif old_files:
        traces_file = old_files[0]
        print(f"\n📂 Cargando: {traces_file}")
        
        try:
            traces = load_traces(traces_file)
            print(f"✅ Cargado exitosamente: {len(traces)} traces")
            analyze_dataset_statistics(traces)
            if traces:
                analyze_trace(traces[0], 0)
        except Exception as e:
            print(f"❌ Error al cargar los traces: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("✅ Análisis completado")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
