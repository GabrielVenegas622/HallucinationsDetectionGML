#!/usr/bin/env python3
"""
Visualización de Grafos de Atención

Este script genera una ilustración del grafo que se construye a partir de
los hidden states y attentions de un trace.

Características:
- Muestra el prompt y la respuesta generada
- Visualiza los nodos (tokens) con sus etiquetas
- Colorea los arcos según la intensidad de atención
- Permite seleccionar qué capa visualizar

Uso:
    python visualize_attention_graph.py \
        --data-pattern "traces_data/*.pkl*" \
        --trace-idx 0 \
        --layer-idx 15 \
        --output graph_visualization.png
"""

import pickle
import gzip
import glob
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import argparse
from pathlib import Path


def load_trace(file_pattern, trace_idx=0):
    """Carga un trace específico de los archivos .pkl/.pkl.gz"""
    files = glob.glob(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos con patrón: {file_pattern}")
    
    # Cargar el primer archivo
    file_path = files[0]
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rb') as f:
            traces = pickle.load(f)
    else:
        with open(file_path, 'rb') as f:
            traces = pickle.load(f)
    
    if trace_idx >= len(traces):
        raise IndexError(f"trace_idx {trace_idx} fuera de rango. Solo hay {len(traces)} traces.")
    
    return traces[trace_idx]


def create_attention_graph(trace, layer_idx, attn_threshold=0.01):
    """
    Crea un grafo de NetworkX a partir de un trace y una capa específica.
    
    Simula exactamente lo que hace el dataloader.
    """
    # 1. Nodos (tokens)
    hidden_states = trace['hidden_states'][layer_idx]
    num_nodes = hidden_states.shape[0]
    
    # 2. Atenciones
    attentions = trace['attentions'][layer_idx]
    
    # Promediar cabezas de atención
    attn_avg = attentions.mean(axis=0)
    
    # Recortar si es necesario (por si hay mismatch)
    if attn_avg.shape[0] > num_nodes or attn_avg.shape[1] > num_nodes:
        attn_avg = attn_avg[:num_nodes, :num_nodes]
    
    # 3. Crear grafo dirigido
    G = nx.DiGraph()
    
    # Agregar nodos con sus etiquetas (tokens)
    tokens_decoded = trace.get('tokens_decoded', [])
    for i in range(num_nodes):
        token_text = tokens_decoded[i] if i < len(tokens_decoded) else f"Token_{i}"
        # Limpiar el texto del token
        token_text = token_text.replace('\n', '\\n').strip()
        if len(token_text) > 15:
            token_text = token_text[:12] + "..."
        G.add_node(i, label=token_text)
    
    # 4. Agregar arcos basados en threshold de atención
    edge_weights = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = attn_avg[i, j]
            if weight > attn_threshold:
                G.add_edge(j, i, weight=weight)  # j -> i (source -> target)
                edge_weights.append(weight)
    
    return G, edge_weights, attn_avg


def visualize_graph(G, edge_weights, trace, layer_idx, output_file='graph_viz.png', 
                   layout='spring', max_nodes_display=50):
    """
    Visualiza el grafo con nodos etiquetados y arcos coloreados por intensidad.
    """
    num_nodes = len(G.nodes())
    
    # Si hay muchos nodos, usar un subset para visualización
    if num_nodes > max_nodes_display:
        print(f"⚠️  El grafo tiene {num_nodes} nodos. Mostrando solo los primeros {max_nodes_display}.")
        nodes_to_show = list(range(max_nodes_display))
        G = G.subgraph(nodes_to_show).copy()
        # Recalcular edge_weights
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Configurar figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                    gridspec_kw={'height_ratios': [1, 4]})
    
    # =========================================================================
    # Panel superior: Información del trace
    # =========================================================================
    ax1.axis('off')
    
    question_id = trace.get('question_id', 'Unknown')
    generated_answer = trace.get('generated_answer_clean', 'Unknown')
    tokens_decoded = trace.get('tokens_decoded', [])
    
    # Reconstruir el prompt (aproximado)
    prompt_text = ' '.join(tokens_decoded[:len(tokens_decoded)//2]) if tokens_decoded else "Prompt"
    response_text = generated_answer
    
    info_text = (
        f"Question ID: {question_id}\n"
        f"Layer: {layer_idx} / {len(trace['hidden_states'])-1}\n"
        f"Total Tokens: {num_nodes}\n"
        f"Edges (connections): {len(G.edges())}\n\n"
        f"Prompt: {prompt_text[:200]}...\n\n"
        f"Response: {response_text[:200]}..."
    )
    
    ax1.text(0.05, 0.5, info_text, 
             fontsize=10, verticalalignment='center',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # =========================================================================
    # Panel inferior: Grafo
    # =========================================================================
    
    # Layout del grafo
    if layout == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:  # sequential (en línea)
        pos = {i: (i, 0) for i in G.nodes()}
    
    # Dibujar nodos
    node_labels = nx.get_node_attributes(G, 'label')
    
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=800,
                          alpha=0.9,
                          ax=ax2)
    
    # Dibujar etiquetas de nodos
    nx.draw_networkx_labels(G, pos, 
                           labels=node_labels,
                           font_size=8,
                           font_family='monospace',
                           ax=ax2)
    
    # Normalizar pesos de arcos para colormap
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        
        # Crear colormap (azul claro -> rojo oscuro)
        cmap = LinearSegmentedColormap.from_list(
            'attention', 
            ['#d0e1f9', '#4d94ff', '#ff6b35', '#ff0000']
        )
        
        # Normalizar pesos
        if max_weight > min_weight:
            norm_weights = [(w - min_weight) / (max_weight - min_weight) 
                           for w in edge_weights]
        else:
            norm_weights = [0.5] * len(edge_weights)
        
        # Colorear arcos según peso
        edge_colors = [cmap(w) for w in norm_weights]
        
        # Dibujar arcos
        nx.draw_networkx_edges(G, pos,
                              edge_color=edge_colors,
                              width=2,
                              alpha=0.6,
                              arrows=True,
                              arrowsize=15,
                              arrowstyle='->',
                              connectionstyle='arc3,rad=0.1',
                              ax=ax2)
        
        # Colorbar para mostrar escala de atención
        sm = plt.cm.ScalarMappable(cmap=cmap, 
                                   norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, orientation='horizontal', 
                           pad=0.05, shrink=0.6)
        cbar.set_label('Attention Weight', fontsize=10)
    else:
        # Sin arcos
        ax2.text(0.5, 0.5, 'No edges above threshold', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=14, color='red')
    
    ax2.set_title(f'Attention Graph - Layer {layer_idx}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.axis('off')
    
    # Guardar
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado en: {output_file}")
    plt.close()


def create_layerwise_comparison(trace, layers_to_show=[0, 15, 31], 
                                output_file='layerwise_comparison.png',
                                attn_threshold=0.01):
    """
    Crea una comparación de grafos entre diferentes capas.
    """
    num_layers_to_show = len(layers_to_show)
    fig, axes = plt.subplots(1, num_layers_to_show, figsize=(6*num_layers_to_show, 6))
    
    if num_layers_to_show == 1:
        axes = [axes]
    
    for idx, layer_idx in enumerate(layers_to_show):
        ax = axes[idx]
        
        # Crear grafo para esta capa
        G, edge_weights, attn_avg = create_attention_graph(trace, layer_idx, attn_threshold)
        
        # Layout simple (circular para claridad)
        pos = nx.circular_layout(G)
        
        # Nodos
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=300, alpha=0.9, ax=ax)
        
        # Etiquetas (solo índices para claridad)
        labels = {i: str(i) for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
        
        # Arcos
        if edge_weights:
            cmap = LinearSegmentedColormap.from_list('attention', 
                                                    ['lightgray', 'blue', 'red'])
            min_w, max_w = min(edge_weights), max(edge_weights)
            norm_weights = [(w - min_w) / (max_w - min_w) if max_w > min_w else 0.5 
                           for w in edge_weights]
            edge_colors = [cmap(w) for w in norm_weights]
            
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                                  width=1.5, alpha=0.5, arrows=True, 
                                  arrowsize=10, ax=ax)
        
        ax.set_title(f'Layer {layer_idx}\n{len(G.edges())} edges', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
    
    question_id = trace.get('question_id', 'Unknown')
    fig.suptitle(f'Layerwise Attention Graph Comparison - {question_id}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Comparación guardada en: {output_file}")
    plt.close()


def create_attention_matrix_heatmap(trace, layer_idx, output_file='attention_heatmap.png'):
    """
    Crea un heatmap de la matriz de atención.
    """
    attentions = trace['attentions'][layer_idx]
    attn_avg = attentions.mean(axis=0)  # Promediar heads
    
    tokens_decoded = trace.get('tokens_decoded', [])
    num_tokens = min(len(tokens_decoded), attn_avg.shape[0])
    
    # Recortar si es necesario
    attn_avg = attn_avg[:num_tokens, :num_tokens]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap
    im = ax.imshow(attn_avg, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Etiquetas de ejes (cada 5 tokens para no saturar)
    step = max(1, num_tokens // 20)
    tick_positions = list(range(0, num_tokens, step))
    tick_labels = [tokens_decoded[i][:10] if i < len(tokens_decoded) else str(i) 
                  for i in tick_positions]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)
    
    ax.set_xlabel('Key Tokens', fontsize=12)
    ax.set_ylabel('Query Tokens', fontsize=12)
    ax.set_title(f'Attention Matrix - Layer {layer_idx}\n'
                f'Question: {trace.get("question_id", "Unknown")}', 
                fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap guardado en: {output_file}")
    plt.close()


def main(args):
    """Función principal"""
    print("="*80)
    print("VISUALIZACIÓN DE GRAFOS DE ATENCIÓN")
    print("="*80)
    
    # Cargar trace
    print(f"\n📂 Cargando trace {args.trace_idx} de {args.data_pattern}...")
    trace = load_trace(args.data_pattern, args.trace_idx)
    
    num_layers = len(trace['hidden_states'])
    print(f"✓ Trace cargado: {trace.get('question_id', 'Unknown')}")
    print(f"  Número de capas: {num_layers}")
    print(f"  Tokens: {len(trace.get('tokens_decoded', []))}")
    print(f"  Respuesta: {trace.get('generated_answer_clean', 'Unknown')[:100]}...")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Visualización principal de una capa
    if args.layer_idx is not None:
        print(f"\n🎨 Generando visualización de capa {args.layer_idx}...")
        G, edge_weights, attn_avg = create_attention_graph(
            trace, args.layer_idx, args.attn_threshold
        )
        
        output_file = output_dir / f"graph_layer_{args.layer_idx}.png"
        visualize_graph(G, edge_weights, trace, args.layer_idx, 
                       output_file, layout=args.layout, 
                       max_nodes_display=args.max_nodes)
    
    # 2. Comparación entre capas
    if args.compare_layers:
        print(f"\n🎨 Generando comparación entre capas...")
        layers_to_compare = [0, num_layers//2, num_layers-1]
        output_file = output_dir / "layerwise_comparison.png"
        create_layerwise_comparison(trace, layers_to_compare, output_file, 
                                   args.attn_threshold)
    
    # 3. Heatmap de matriz de atención
    if args.create_heatmap:
        layer_for_heatmap = args.layer_idx if args.layer_idx is not None else num_layers//2
        print(f"\n🎨 Generando heatmap de atención de capa {layer_for_heatmap}...")
        output_file = output_dir / f"attention_heatmap_layer_{layer_for_heatmap}.png"
        create_attention_matrix_heatmap(trace, layer_for_heatmap, output_file)
    
    print("\n" + "="*80)
    print("✅ VISUALIZACIÓN COMPLETADA")
    print(f"📁 Archivos guardados en: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualizar grafos de atención de traces"
    )
    
    parser.add_argument('--data-pattern', type=str, required=True,
                       help='Patrón glob para archivos .pkl')
    parser.add_argument('--trace-idx', type=int, default=0,
                       help='Índice del trace a visualizar')
    parser.add_argument('--layer-idx', type=int, default=15,
                       help='Índice de la capa a visualizar (default: 15)')
    parser.add_argument('--attn-threshold', type=float, default=0.01,
                       help='Umbral de atención para crear arcos')
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                       help='Directorio de salida para las imágenes')
    parser.add_argument('--layout', type=str, default='spring',
                       choices=['spring', 'circular', 'kamada_kawai', 'sequential'],
                       help='Layout del grafo')
    parser.add_argument('--max-nodes', type=int, default=50,
                       help='Máximo número de nodos a mostrar')
    parser.add_argument('--compare-layers', action='store_true',
                       help='Crear comparación entre capas')
    parser.add_argument('--create-heatmap', action='store_true',
                       help='Crear heatmap de matriz de atención')
    
    args = parser.parse_args()
    
    main(args)
