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
    print(f"\n💬 Respuesta Generada: {trace['generated_answer']}")
    
    if trace['ground_truth_answers']:
        print(f"\n✅ Respuestas Correctas:")
        for i, ans in enumerate(trace['ground_truth_answers'][:3], 1):
            print(f"   {i}. {ans}")
    
    # Información de tokens
    num_tokens_total = len(trace['tokens'])
    num_tokens_generated = num_tokens_total - trace['prompt_length']
    print(f"\n🔢 Tokens:")
    print(f"   - Total: {num_tokens_total}")
    print(f"   - Prompt: {trace['prompt_length']}")
    print(f"   - Generados: {num_tokens_generated}")
    
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
    for trace in traces:
        num_generated = len(trace['tokens']) - trace['prompt_length']
        answer_lengths.append(num_generated)
    
    print(f"\n📏 Longitud de respuestas generadas:")
    print(f"   - Media: {np.mean(answer_lengths):.2f} tokens")
    print(f"   - Mediana: {np.median(answer_lengths):.2f} tokens")
    print(f"   - Min: {np.min(answer_lengths)} tokens")
    print(f"   - Max: {np.max(answer_lengths)} tokens")
    print(f"   - Std: {np.std(answer_lengths):.2f} tokens")
    
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
    
    # Buscar archivos pickle
    pickle_files = list(traces_dir.glob("*.pkl"))
    
    if not pickle_files:
        print(f"❌ No se encontraron archivos .pkl en {traces_dir}")
        return
    
    print(f"✅ Archivos encontrados:")
    for i, f in enumerate(pickle_files, 1):
        print(f"   {i}. {f.name}")
    
    # Cargar el primer archivo
    traces_file = pickle_files[0]
    print(f"\n📂 Cargando: {traces_file}")
    
    try:
        traces = load_traces(traces_file)
        print(f"✅ Cargado exitosamente: {len(traces)} traces")
        
        # Estadísticas globales
        analyze_dataset_statistics(traces)
        
        # Analizar primer ejemplo
        if traces:
            analyze_trace(traces[0], 0)
        
        # Analizar algunos ejemplos más
        if len(traces) > 1:
            print(f"\n\n{'='*80}")
            print("EJEMPLOS ADICIONALES (resumen)")
            print(f"{'='*80}")
            
            for idx in [1, 2, 3]:
                if idx < len(traces):
                    trace = traces[idx]
                    num_gen = len(trace['tokens']) - trace['prompt_length']
                    print(f"\nTrace #{idx}:")
                    print(f"  Q: {trace['question'][:60]}...")
                    print(f"  A: {trace['generated_answer'][:60]}...")
                    print(f"  Tokens generados: {num_gen}")
        
        print(f"\n{'='*80}")
        print("✅ Análisis completado")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Error al cargar los traces: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
