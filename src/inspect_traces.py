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


def analyze_trace(trace, trace_idx=0, dataset=None):
    """Analiza un trace individual y muestra información detallada."""
    print(f"\n{'='*80}")
    print(f"ANÁLISIS DEL TRACE #{trace_idx}")
    print(f"{'='*80}")
    
    # Información básica
    question_id = trace.get('question_id', 'N/A')
    print(f"\n🆔 Question ID: {question_id}")
    
    # Si tenemos acceso al dataset, recuperar pregunta y respuestas
    if dataset is not None and question_id != 'N/A':
        try:
            # Buscar en el dataset por question_id
            example = next((ex for ex in dataset if ex['question_id'] == question_id), None)
            if example:
                print(f"\n📝 Pregunta (recuperada): {example['question']}")
                if 'answer' in example:
                    print(f"\n✅ Respuestas Correctas:")
                    for i, ans in enumerate(example['answer']['aliases'][:3], 1):
                        print(f"   {i}. {ans}")
        except:
            pass
    
    print(f"\n💬 Respuesta Generada: {trace['generated_answer_clean']}")
    
    # Información de tokens
    num_tokens_generated = len(trace['tokens'])
    print(f"\n🔢 Tokens:")
    print(f"   - Tokens en respuesta: {num_tokens_generated}")
    
    # Mostrar tokens decodificados si están disponibles
    if 'tokens_decoded' in trace:
        print(f"\n📝 Tokens decodificados:")
        for i, token_text in enumerate(trace['tokens_decoded'][:10]):
            print(f"   {i}: '{token_text}'")
    
    # Información de capas
    num_layers = len(trace['hidden_states'])
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
        # Tokens solo contiene la respuesta limpia
        num_generated = len(trace['tokens'])
        answer_lengths.append(num_generated)
    
    print(f"\n📏 Longitud de respuestas limpias:")
    print(f"   - Media: {np.mean(answer_lengths):.2f} tokens")
    print(f"   - Mediana: {np.median(answer_lengths):.2f} tokens")
    print(f"   - Min: {np.min(answer_lengths)} tokens")
    print(f"   - Max: {np.max(answer_lengths)} tokens")
    print(f"   - Std: {np.std(answer_lengths):.2f} tokens")
    
    # Verificar consistencia de capas
    num_layers_list = [len(trace['hidden_states']) for trace in traces]
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
    
    # Intentar cargar TriviaQA para recuperar información
    dataset = None
    try:
        from datasets import load_dataset
        print("Cargando TriviaQA para recuperar preguntas y respuestas...")
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
        print(f"✅ Dataset cargado: {len(dataset)} ejemplos")
    except Exception as e:
        print(f"⚠️  No se pudo cargar TriviaQA: {e}")
        print("   Continuando sin recuperar preguntas/respuestas originales")
    
    # Buscar archivos pickle (batch y archivos antiguos)
    batch_files = sorted(traces_dir.glob("trivia_qa_traces_batch_*.pkl"))
    old_files = list(traces_dir.glob("trivia_qa_traces_*.pkl"))
    old_files = [f for f in old_files if "batch" not in f.name]
    
    if not batch_files and not old_files:
        print(f"❌ No se encontraron archivos .pkl en {traces_dir}")
        return
    
    # Mostrar información sobre archivos encontrados
    print(f"\n{'='*80}")
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
        
        # Analizar cada batch y mostrar 5 ejemplos
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
                
                # Mostrar 5 ejemplos de este batch
                print(f"\n   --- 5 Ejemplos del batch {batch_idx} ---")
                for i in range(min(5, len(traces))):
                    trace = traces[i]
                    question_id = trace.get('question_id', 'N/A')
                    num_tokens = len(trace['tokens'])
                    
                    # Intentar recuperar pregunta del dataset
                    question_text = "N/A"
                    if dataset is not None:
                        try:
                            example = next((ex for ex in dataset if ex['question_id'] == question_id), None)
                            if example:
                                question_text = example['question'][:60] + "..."
                        except:
                            pass
                    
                    print(f"\n   {i+1}. Question ID: {question_id}")
                    print(f"      Pregunta: {question_text}")
                    print(f"      Respuesta: {trace['generated_answer_clean'][:60]}...")
                    print(f"      Tokens: {num_tokens}")
                    if 'tokens_decoded' in trace:
                        print(f"      Decodificados: {trace['tokens_decoded'][:3]}...")
                
                # Solo mostrar detalle completo del primer ejemplo del primer batch
                if batch_idx == 0 and traces:
                    print(f"\n{'='*80}")
                    print("ANÁLISIS DETALLADO DEL PRIMER TRACE")
                    print(f"{'='*80}")
                    analyze_trace(traces[0], 0, dataset)
                
            except Exception as e:
                print(f"   ❌ Error cargando batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
        
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
    
    # Si hay archivos antiguos, también analizarlos
    elif old_files:
        traces_file = old_files[0]
        print(f"\n📂 Cargando: {traces_file}")
        
        try:
            traces = load_traces(traces_file)
            print(f"✅ Cargado exitosamente: {len(traces)} traces")
            analyze_dataset_statistics(traces)
            if traces:
                analyze_trace(traces[0], 0, dataset)
        except Exception as e:
            print(f"❌ Error al cargar los traces: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("✅ Análisis completado")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
