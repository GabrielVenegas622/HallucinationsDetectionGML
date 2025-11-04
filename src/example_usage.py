"""
Ejemplo de cómo cargar y usar los traces extraídos.
Este script demuestra cómo trabajar con los datos guardados.
"""

import pickle
import numpy as np
from pathlib import Path


def load_and_explore():
    """Ejemplo completo de carga y exploración de datos."""
    
    # 1. Cargar los datos
    traces_file = Path("./traces_data/trivia_qa_traces_Qwen3-4B-Instruct-2507.pkl")
    
    if not traces_file.exists():
        print("❌ Archivo de traces no encontrado.")
        print("   Ejecuta primero: python src/trace_extractor.py")
        return
    
    print("📂 Cargando traces...")
    with open(traces_file, 'rb') as f:
        all_traces = pickle.load(f)
    
    print(f"✅ Cargados {len(all_traces)} ejemplos\n")
    
    # 2. Explorar un ejemplo
    example = all_traces[0]
    
    print("="*80)
    print("ESTRUCTURA DE UN TRACE")
    print("="*80)
    print(f"\nClaves disponibles: {list(example.keys())}\n")
    
    # 3. Acceder a datos específicos
    print("📝 Pregunta:")
    print(f"   {example['question']}\n")
    
    print("💬 Respuesta Generada:")
    print(f"   {example['generated_answer']}\n")
    
    print("✅ Respuestas Correctas:")
    for i, ans in enumerate(example['ground_truth_answers'][:3], 1):
        print(f"   {i}. {ans}")
    
    # 4. Trabajar con Hidden States
    print("\n" + "="*80)
    print("HIDDEN STATES (ACTIVACIONES)")
    print("="*80)
    
    hidden_states = example['hidden_states']  # [num_layers][num_tokens]
    num_layers = len(hidden_states)
    
    print(f"\n📊 Estructura:")
    print(f"   - Número de capas: {num_layers}")
    print(f"   - Tokens generados por capa: {len(hidden_states[0])}")
    
    # Ejemplo: Extraer activaciones de la capa 10, token 5
    layer_idx = 10
    token_idx = 5
    
    if layer_idx < num_layers and token_idx < len(hidden_states[layer_idx]):
        activation = hidden_states[layer_idx][token_idx]
        print(f"\n🔍 Ejemplo - Activación en capa {layer_idx}, token {token_idx}:")
        print(f"   - Shape: {activation.shape}")
        print(f"   - Media: {activation.mean():.4f}")
        print(f"   - Std: {activation.std():.4f}")
    
    # 5. Trabajar con Attention Matrices
    print("\n" + "="*80)
    print("ATTENTION MATRICES (ATENCIONES)")
    print("="*80)
    
    attentions = example['attentions']  # [num_layers][num_tokens]
    
    print(f"\n📊 Estructura:")
    print(f"   - Número de capas: {len(attentions)}")
    print(f"   - Tokens generados por capa: {len(attentions[0])}")
    
    # Ejemplo: Extraer matriz de atención de la capa 10, token 5
    if layer_idx < len(attentions) and token_idx < len(attentions[layer_idx]):
        attn_matrix = attentions[layer_idx][token_idx]
        batch, num_heads, seq_len, _ = attn_matrix.shape
        
        print(f"\n🔍 Ejemplo - Atención en capa {layer_idx}, token {token_idx}:")
        print(f"   - Shape: {attn_matrix.shape}")
        print(f"   - Batch size: {batch}")
        print(f"   - Número de cabezas: {num_heads}")
        print(f"   - Longitud de secuencia: {seq_len}")
        
        # Promediar sobre cabezas para obtener matriz de atención global
        avg_attn = attn_matrix[0].mean(axis=0)  # [seq_len, seq_len]
        print(f"\n   📈 Matriz promedio (sobre cabezas):")
        print(f"      - Shape: {avg_attn.shape}")
        print(f"      - Suma por fila (debe ser ~1): {avg_attn.sum(axis=1)[:3]}")
    
    # 6. Ejemplo: Iterar sobre todas las capas
    print("\n" + "="*80)
    print("EJEMPLO: ITERAR SOBRE CAPAS")
    print("="*80)
    
    print("\nActivaciones en el primer token generado para cada capa:")
    for layer_idx in range(min(5, num_layers)):  # Primeras 5 capas
        if hidden_states[layer_idx]:
            first_token_activation = hidden_states[layer_idx][0]
            mean_activation = first_token_activation.mean()
            print(f"   Capa {layer_idx:2d}: media = {mean_activation:.4f}")
    
    # 7. Ejemplo: Construir grafo simple desde matriz de atención
    print("\n" + "="*80)
    print("EJEMPLO: CONSTRUCCIÓN DE GRAFO (CONCEPTUAL)")
    print("="*80)
    
    # Tomamos una matriz de atención de ejemplo
    example_attn = attentions[15][0]  # Capa 15, primer token generado
    avg_attn = example_attn[0].mean(axis=0)  # Promediar sobre cabezas
    
    print(f"\n🕸️  Grafo desde matriz de atención:")
    print(f"   - Nodos: {avg_attn.shape[0]} (cada token es un nodo)")
    print(f"   - Aristas: potencialmente {avg_attn.shape[0]}² = {avg_attn.shape[0]**2}")
    
    # Ejemplo: Filtrar aristas con peso > threshold
    threshold = 0.1
    strong_connections = (avg_attn > threshold).sum()
    print(f"   - Conexiones fuertes (peso > {threshold}): {strong_connections}")
    
    # 8. Ejemplo: Procesar múltiples ejemplos
    print("\n" + "="*80)
    print("EJEMPLO: PROCESAR BATCH DE EJEMPLOS")
    print("="*80)
    
    print("\nPrimeros 5 ejemplos del dataset:")
    for i in range(min(5, len(all_traces))):
        trace = all_traces[i]
        num_tokens = len(trace['tokens']) - trace['prompt_length']
        print(f"\n{i+1}. Pregunta: {trace['question'][:50]}...")
        print(f"   Respuesta: {trace['generated_answer'][:50]}...")
        print(f"   Tokens generados: {num_tokens}")
    
    print("\n" + "="*80)
    print("✅ EXPLORACIÓN COMPLETADA")
    print("="*80)
    print("\nPróximos pasos:")
    print("  1. Implementar construcción de grafos desde matrices de atención")
    print("  2. Definir DataLoader para PyTorch Geometric")
    print("  3. Implementar VAE para grafos")
    print("")


if __name__ == "__main__":
    load_and_explore()
