"""
Script de ejemplo que muestra cómo usar tokens decodificados 
para visualización de grafos de atención.
"""

import pickle
import numpy as np
from pathlib import Path


def visualize_attention_graph_example():
    """
    Demuestra cómo usar tokens_decoded para visualizar grafos.
    """
    
    print("="*80)
    print("EJEMPLO: VISUALIZACIÓN DE GRAFOS CON TOKENS DECODIFICADOS")
    print("="*80)
    
    # Simular un trace con tokens decodificados
    example_trace = {
        'question': 'What is the capital of France?',
        'generated_answer_clean': 'Paris.',
        'tokens': np.array([12345, 67890, 11111]),  # IDs (no legibles)
        'tokens_decoded': ['Paris', '.', ''],  # Textos (legibles)
        'attentions': None,  # Simularemos una matriz
        'num_layers': 36
    }
    
    print("\n1️⃣  DATOS DEL TRACE:")
    print(f"   Pregunta: {example_trace['question']}")
    print(f"   Respuesta: {example_trace['generated_answer_clean']}")
    print(f"   Número de tokens: {len(example_trace['tokens'])}")
    
    print("\n2️⃣  COMPARACIÓN: IDs vs DECODIFICADOS:")
    print("\n   Token IDs (no legibles):")
    for i, token_id in enumerate(example_trace['tokens']):
        print(f"      Token {i}: {token_id}")
    
    print("\n   Tokens decodificados (legibles):")
    for i, token_text in enumerate(example_trace['tokens_decoded']):
        print(f"      Token {i}: '{token_text}'")
    
    print("\n3️⃣  CONSTRUCCIÓN DE GRAFO:")
    print("\n   Sin tokens_decoded:")
    print("      Nodos: [12345, 67890, 11111]")
    print("      Visualización: Números sin significado")
    
    print("\n   Con tokens_decoded:")
    print("      Nodos: ['Paris', '.', '']")
    print("      Visualización: Texto legible para humanos")
    
    # Simular matriz de atención
    num_tokens = len(example_trace['tokens_decoded'])
    attn_matrix = np.random.rand(num_tokens, num_tokens)
    attn_matrix = attn_matrix / attn_matrix.sum(axis=1, keepdims=True)  # Normalizar
    
    print("\n4️⃣  MATRIZ DE ATENCIÓN (simulada):")
    print("\n   Formato con IDs:")
    print("        12345   67890   11111")
    print(f"   12345 {attn_matrix[0, 0]:.3f}   {attn_matrix[0, 1]:.3f}   {attn_matrix[0, 2]:.3f}")
    print(f"   67890 {attn_matrix[1, 0]:.3f}   {attn_matrix[1, 1]:.3f}   {attn_matrix[1, 2]:.3f}")
    print(f"   11111 {attn_matrix[2, 0]:.3f}   {attn_matrix[2, 1]:.3f}   {attn_matrix[2, 2]:.3f}")
    
    print("\n   Formato con tokens decodificados:")
    print("        'Paris' '.'     ''")
    print(f"   'Paris' {attn_matrix[0, 0]:.3f}   {attn_matrix[0, 1]:.3f}   {attn_matrix[0, 2]:.3f}")
    print(f"   '.'     {attn_matrix[1, 0]:.3f}   {attn_matrix[1, 1]:.3f}   {attn_matrix[1, 2]:.3f}")
    print(f"   ''      {attn_matrix[2, 0]:.3f}   {attn_matrix[2, 1]:.3f}   {attn_matrix[2, 2]:.3f}")
    
    print("\n5️⃣  INTERPRETACIÓN:")
    print(f"\n   El token 'Paris' presta atención a:")
    print(f"      - Sí mismo ('Paris'): {attn_matrix[0, 0]:.1%}")
    print(f"      - Al punto ('.'): {attn_matrix[0, 1]:.1%}")
    print(f"      - Al token vacío (''): {attn_matrix[0, 2]:.1%}")


def example_networkx_visualization():
    """
    Ejemplo de cómo crear visualización con NetworkX.
    """
    
    print("\n\n" + "="*80)
    print("EJEMPLO: CÓDIGO PARA VISUALIZACIÓN CON NETWORKX")
    print("="*80)
    
    code = '''
import networkx as nx
import matplotlib.pyplot as plt

# Cargar trace
trace = batch[0]
tokens_decoded = trace['tokens_decoded']
attn_matrix = trace['attentions'][15][0][0].mean(axis=0)  # Capa 15, promediar cabezas

# Crear grafo dirigido
G = nx.DiGraph()

# Añadir nodos con labels legibles
for i, token_text in enumerate(tokens_decoded):
    G.add_node(i, label=token_text)

# Añadir aristas con pesos de atención
threshold = 0.1  # Solo aristas con peso > 0.1
for i in range(len(tokens_decoded)):
    for j in range(len(tokens_decoded)):
        if attn_matrix[i, j] > threshold:
            G.add_edge(i, j, weight=attn_matrix[i, j])

# Visualizar
pos = nx.spring_layout(G)
labels = {i: trace['tokens_decoded'][i] for i in G.nodes()}

plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')
nx.draw_networkx_labels(G, pos, labels, font_size=12)

# Mostrar pesos de aristas
edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" 
               for i, j in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

plt.title(f"Grafo de Atención: {trace['question']}")
plt.axis('off')
plt.tight_layout()
plt.savefig('attention_graph.png', dpi=300, bbox_inches='tight')
plt.show()
'''
    
    print("\n" + code)
    
    print("\n✅ Ventajas de usar tokens_decoded:")
    print("   • Labels legibles para humanos")
    print("   • Fácil interpretación de patrones")
    print("   • Depuración más simple")
    print("   • Presentaciones y papers más claros")


def example_pyg_graph_construction():
    """
    Ejemplo de cómo construir grafos para PyTorch Geometric.
    """
    
    print("\n\n" + "="*80)
    print("EJEMPLO: CONSTRUCCIÓN DE GRAFOS PARA PYTORCH GEOMETRIC")
    print("="*80)
    
    code = '''
import torch
from torch_geometric.data import Data

def build_graph_from_trace(trace, layer_idx=15, head_avg=True):
    """
    Construye un objeto Data de PyTorch Geometric desde un trace.
    
    Args:
        trace: Diccionario con los datos del trace
        layer_idx: Índice de la capa a usar
        head_avg: Si True, promedia sobre cabezas de atención
    
    Returns:
        torch_geometric.data.Data
    """
    # Obtener datos
    tokens_decoded = trace['tokens_decoded']
    num_tokens = len(tokens_decoded)
    
    # Hidden states como features de nodos [num_tokens, hidden_dim]
    hidden_states = trace['hidden_states'][layer_idx]
    node_features = torch.tensor(
        np.stack([h[0, -1, :] for h in hidden_states]),  # Último token de cada paso
        dtype=torch.float
    )
    
    # Atención como aristas [num_tokens, num_tokens]
    attn = trace['attentions'][layer_idx][0][0]  # Primera generación
    if head_avg:
        attn = attn.mean(axis=0)  # Promediar sobre cabezas
    
    # Construir lista de aristas (edge_index)
    threshold = 0.1
    edge_index = []
    edge_attr = []
    
    for i in range(num_tokens):
        for j in range(num_tokens):
            if attn[i, j] > threshold:
                edge_index.append([i, j])
                edge_attr.append(attn[i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
    
    # Crear objeto Data
    graph = Data(
        x=node_features,           # [num_nodes, num_features]
        edge_index=edge_index,     # [2, num_edges]
        edge_attr=edge_attr,       # [num_edges, 1]
        num_nodes=num_tokens,
        
        # Metadata útil
        tokens_decoded=tokens_decoded,  # Para visualización
        tokens_ids=trace['tokens'],     # IDs originales
        question=trace['question'],
        answer=trace['generated_answer_clean']
    )
    
    return graph

# Uso
from src.batch_loader import TraceBatchLoader

loader = TraceBatchLoader()
trace = loader.get_batch(0)[0]

graph = build_graph_from_trace(trace, layer_idx=15)

print(f"Grafo construido:")
print(f"  - Nodos: {graph.num_nodes}")
print(f"  - Aristas: {graph.edge_index.shape[1]}")
print(f"  - Features por nodo: {graph.x.shape[1]}")
print(f"  - Tokens: {graph.tokens_decoded}")
'''
    
    print("\n" + code)
    
    print("\n✅ Información disponible en el grafo:")
    print("   • x: Features de nodos (hidden states)")
    print("   • edge_index: Conectividad (de matriz de atención)")
    print("   • edge_attr: Pesos de aristas (valores de atención)")
    print("   • tokens_decoded: Labels legibles ★")
    print("   • tokens_ids: IDs originales (para procesar)")


def show_real_example_output():
    """
    Muestra cómo se vería la salida real de un trace.
    """
    
    print("\n\n" + "="*80)
    print("EJEMPLO: SALIDA REAL DE UN TRACE")
    print("="*80)
    
    print("""
Estructura de cada trace guardado:
{
    'question': 'What is the capital of France?',
    'generated_answer_clean': 'Paris.',
    
    'tokens': array([12345, 67890]),  # IDs numéricos
    'tokens_decoded': ['Paris', '.'],  # ★ Textos legibles
    
    'hidden_states': [
        [array([...]), array([...])],  # Capa 0: 2 tokens
        [array([...]), array([...])],  # Capa 1: 2 tokens
        ...                            # 36 capas total
    ],
    
    'attentions': [
        [array([...]), array([...])],  # Capa 0: 2 matrices
        [array([...]), array([...])],  # Capa 1: 2 matrices
        ...                            # 36 capas total
    ],
    
    'num_layers': 36,
    'prompt_length': 15,
    'cutoff_method': 'first_period',
    'tokens_before_cutoff': 2,
    'tokens_after_cutoff': 33,
    
    # Metadata
    'example_id': 0,
    'global_example_id': 0,
    'batch_number': 0,
    'ground_truth_answers': ['Paris', 'Paris, France']
}
    """)
    
    print("\n✨ DIFERENCIA CLAVE:")
    print("\n   Antes (sin tokens_decoded):")
    print("      - Visualización: Nodo 12345 → Nodo 67890")
    print("      - Problema: ¿Qué significan estos números?")
    
    print("\n   Ahora (con tokens_decoded):")
    print("      - Visualización: Nodo 'Paris' → Nodo '.'")
    print("      - Beneficio: ¡Inmediatamente legible!")


if __name__ == "__main__":
    visualize_attention_graph_example()
    example_networkx_visualization()
    example_pyg_graph_construction()
    show_real_example_output()
    
    print("\n" + "="*80)
    print("✅ RESUMEN")
    print("="*80)
    print("""
Los tokens_decoded permiten:
  1. Visualización legible de grafos
  2. Debugging más fácil
  3. Presentaciones claras
  4. Análisis interpretable
  5. Validación visual de patrones

Los datos guardados contienen:
  • Solo respuesta limpia (sin redundancia)
  • Tokens como IDs (para procesamiento)
  • Tokens como texto (para visualización) ★
  • Trazas completas hasta el punto de corte

Próximo paso:
  → Implementar dataloader que construya grafos PyG
    """)
    print("="*80 + "\n")
