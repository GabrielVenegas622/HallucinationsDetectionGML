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
        # Analizar primera capa (ahora es un array 2D, no una lista)
        first_layer = trace['hidden_states'][0]
        print(f"   - Shape primera capa: {first_layer.shape}")
        seq_len_total, hidden_dim = first_layer.shape
        print(f"   - Tokens totales (prompt + respuesta): {seq_len_total}")
        print(f"   - Dimensión oculta: {hidden_dim}")
    
    # Análisis de atenciones
    print(f"\n👁️  Attention Matrices:")
    print(f"   - Estructura: {len(trace['attentions'])} capas")
    if trace['attentions']:
        # Analizar primera capa (ahora es un array 3D, no una lista)
        first_layer_attn = trace['attentions'][0]
        print(f"   - Shape primera capa: {first_layer_attn.shape}")
        num_heads, seq_len, _ = first_layer_attn.shape
        print(f"   - Número de cabezas: {num_heads}")
        print(f"   - Secuencia total: {seq_len}")
        
        # Estadísticas de la matriz de atención
        print(f"\n   📊 Estadísticas (promedio sobre cabezas):")
        avg_attn = first_layer_attn.mean(axis=0)  # Promediar sobre cabezas
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
    
    # Intentar detectar qué dataset se usó basándose en los nombres de archivo
    all_files = list(traces_dir.glob("*.pkl"))
    
    dataset_name = None
    if any('triviaqa' in f.name.lower() for f in all_files):
        dataset_name = 'triviaqa'
        
    elif any('truthfulqa' in f.name.lower() for f in all_files):
        dataset_name = 'truthfulqa'
    
    print(f'Se ha identificado el dataset: {dataset_name}')
    
    # Intentar cargar dataset para recuperar información
    dataset = None
    if dataset_name:
        try:
            from datasets import load_dataset
            print(f"Cargando {dataset_name} para recuperar preguntas y respuestas...")
            
            if dataset_name == 'triviaqa':
                dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
            elif dataset_name == 'truthfulqa':
                dataset = load_dataset("truthful_qa", "generation", split="validation")
            
            print(f"✅ Dataset cargado: {len(dataset)} ejemplos")
        except Exception as e:
            print(f"⚠️  No se pudo cargar {dataset_name}: {e}")
            print("   Continuando sin recuperar preguntas/respuestas originales")
    
    # Buscar archivos pickle con cualquier patrón
    batch_files = sorted(traces_dir.glob("*_batch_*.pkl"))
    
    if not batch_files:
        print(f"❌ No se encontraron archivos .pkl en {traces_dir}")
        return
    
    # Mostrar información sobre archivos encontrados
    print(f"\n{'='*80}")
    print("ARCHIVOS DE TRACES ENCONTRADOS")
    print(f"{'='*80}\n")
    
    print(f"✅ Archivos en batch: {len(batch_files)}")
    total_size = 0
    for f in batch_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"   • {f.name}: {size_mb:.2f} MB")
    print(f"\n💾 Tamaño total de batches: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    # Cargar y analizar batches
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
            for i in range(min(20, len(traces))):
                trace = traces[i]
                question_id = trace.get('question_id', 'N/A')
                num_tokens = len(trace['tokens'])
                
                # Intentar recuperar pregunta del dataset
                question_text = "N/A"
                if dataset is not None:
                    try:
                        if dataset_name == 'triviaqa':
                            # Buscar por question_id
                            example = next((ex for ex in dataset if ex['question_id'] == question_id), None)
                            answer = example['answer']['normalized_aliases'][:3]
                        elif dataset_name == 'truthfulqa':
                            # Extraer índice del question_id (formato: truthfulqa_123)
                            if question_id.startswith('truthfulqa_'):
                                idx = int(question_id.split('_')[1])
                                example = dataset[idx]
                                answer = None
                            else:
                                example = None
                        else:
                            example = None
                        
                        if example:
                            question_text = example['question'] 
                    except:
                        pass
                
                print(f"\n   {i+1}. Question ID: {question_id}")
                print(f"      Pregunta: {question_text}")
                print(f"      Respuesta: {trace['generated_answer_clean']}")
                print(f"      Ground Truth: {answer}")
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
    
    print(f"\n{'='*80}")
    print("✅ Análisis completado")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
