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
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
import argparse
from pathlib import Path

# Intentar importar graph-tool
try:
    from graph_tool.all import Graph, graph_draw, sfdp_layout, arf_layout, fruchterman_reingold_layout
    import graph_tool.all as gt
    GRAPH_TOOL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: graph-tool no está disponible: {e}")
    print("Por favor, instala graph-tool siguiendo las instrucciones en:")
    print("https://graph-tool.skewed.de/")
    print("\nPara sistemas basados en Arch/Manjaro: pacman -S python-graph-tool")
    print("Para sistemas basados en Debian/Ubuntu: apt-get install python3-graph-tool")
    print("Para conda: conda install -c conda-forge graph-tool")
    GRAPH_TOOL_AVAILABLE = False
    raise


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
    Crea un grafo de graph-tool a partir de un trace y una capa específica.
    
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
    
    # 3. Crear grafo dirigido con graph-tool
    G = Graph(directed=True)
    
    # Agregar nodos
    G.add_vertex(num_nodes)
    
    # Propiedades de nodos y arcos
    v_label = G.new_vertex_property("string")
    e_weight = G.new_edge_property("double")
    
    # Agregar etiquetas a los nodos
    tokens_decoded = trace.get('tokens_decoded', [])
    for i in range(num_nodes):
        token_text = tokens_decoded[i] if i < len(tokens_decoded) else f"Token_{i}"
        # Limpiar el texto del token
        token_text = token_text.replace('\n', '\\n').strip()
        if len(token_text) > 15:
            token_text = token_text[:12] + "..."
        v_label[G.vertex(i)] = token_text
    
    # 4. Agregar arcos basados en threshold de atención
    edge_weights = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = attn_avg[i, j]
            if weight > attn_threshold:
                e = G.add_edge(G.vertex(j), G.vertex(i))  # j -> i (source -> target)
                e_weight[e] = weight
                edge_weights.append(weight)
    
    G.vertex_properties["label"] = v_label
    G.edge_properties["weight"] = e_weight
    
    return G, edge_weights, attn_avg


def visualize_graph(G, edge_weights, trace, layer_idx, output_file='graph_viz.png', 
                   layout='spring', max_nodes_display=50):
    """
    Visualiza el grafo con nodos etiquetados y arcos coloreados por intensidad usando graph-tool.
    Usa 2 colores HLS (posiciones 6 y 7 de 8) para distinguir prompt vs respuesta.
    Arcos usan colormap viridis.
    """
    num_nodes = G.num_vertices()
    original_num_nodes = num_nodes
    
    # Si hay muchos nodos, usar un subset para visualización
    if num_nodes > max_nodes_display:
        print(f"⚠️  El grafo tiene {num_nodes} nodos. Mostrando solo los primeros {max_nodes_display}.")
        # Crear filtro de vértices - IMPORTANTE: mantener las propiedades antes de filtrar
        vfilt = G.new_vertex_property("bool")
        for v in G.vertices():
            vfilt[v] = int(v) < max_nodes_display
        G.set_vertex_filter(vfilt)
        
        # Recalcular edge_weights
        edge_weights = [G.ep.weight[e] for e in G.edges()]
        num_nodes = G.num_vertices()
    
    # =========================================================================
    # Layout del grafo usando graph-tool con SFDP
    # =========================================================================
    print(f"  ➜ Calculando layout SFDP...")
    pos = sfdp_layout(G, cooling_step=0.95, epsilon=1e-2, max_iter=1000)
    
    # =========================================================================
    # Colores de nodos: 2 colores HLS (posiciones 6 y 7 de paleta con 8 colores)
    # =========================================================================
    # Obtener paleta HLS con 8 colores discretos
    hls_palette = sns.color_palette("hls", 8)
    color_prompt = hls_palette[6]    # Morado oscuro (posición 6)
    color_response = hls_palette[7]   # Violeta (posición 7)
    
    # Determinar punto de corte entre prompt y respuesta
    # Asumimos que la primera mitad son tokens del prompt y la segunda de la respuesta
    tokens_decoded = trace.get('tokens_decoded', [])
    total_tokens = len(tokens_decoded)
    
    # Buscar punto de corte más preciso si hay información de separación
    # Por defecto, usamos la mitad
    prompt_end_idx = total_tokens // 2
    
    # Intentar detectar mejor el punto de corte buscando patrones comunes
    # (esto es una heurística, podrías ajustarla según tu formato)
    for i, token in enumerate(tokens_decoded):
        if token.strip().lower() in ['answer:', 'a:', 'response:', '\n\n']:
            prompt_end_idx = i
            break
    
    print(f"  ➜ Punto de corte prompt/respuesta: token {prompt_end_idx}")
    
    vertex_fill_color = G.new_vertex_property("vector<double>")
    for v in G.vertices():
        idx = int(v)
        if idx < prompt_end_idx:
            # Token del prompt - morado oscuro (posición 6)
            r, g, b = color_prompt
        else:
            # Token de la respuesta - violeta (posición 7)
            r, g, b = color_response
        vertex_fill_color[v] = [r, g, b, 0.85]  # Alpha 0.85 para buena visibilidad
    
    # =========================================================================
    # Tamaño uniforme de nodos
    # =========================================================================
    vertex_size = G.new_vertex_property("double")
    for v in G.vertices():
        vertex_size[v] = 40  # Tamaño uniforme para todos los nodos
    
    # =========================================================================
    # Arcos con colormap viridis según peso de atención
    # =========================================================================
    if edge_weights and len(edge_weights) > 0:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        
        # Usar viridis colormap (compatible con versiones antiguas y nuevas)
        try:
            viridis_cmap = cm.get_cmap('viridis')
        except AttributeError:
            import matplotlib.pyplot as plt
            viridis_cmap = plt.get_cmap('viridis')
        
        # Propiedades de visualización de arcos
        edge_pen_width = G.new_edge_property("double")
        edge_color = G.new_edge_property("vector<double>")
        
        for e in G.edges():
            weight = G.ep.weight[e]
            # Normalizar peso entre 0 y 1
            norm_weight = (weight - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5
            
            # Ancho de línea proporcional al peso (entre 1 y 8)
            edge_pen_width[e] = 1 + norm_weight * 7
            
            # Color usando viridis
            rgba = viridis_cmap(norm_weight)
            edge_color[e] = [rgba[0], rgba[1], rgba[2], 0.6 + 0.4 * norm_weight]
    else:
        edge_pen_width = G.new_edge_property("double")
        edge_color = G.new_edge_property("vector<double>")
        for e in G.edges():
            edge_pen_width[e] = 2
            edge_color[e] = [0.5, 0.5, 0.5, 0.5]
    
    # =========================================================================
    # Usar las etiquetas que ya están en el grafo (tokens)
    # =========================================================================
    vertex_text_prop = G.vp.label
    
    # Crear imagen del grafo directamente con graph-tool
    question_id = trace.get('question_id', 'Unknown')
    
    print(f"  ➜ Dibujando grafo con {G.num_vertices()} nodos y {G.num_edges()} aristas...")
    print(f"  ➜ Colores: Prompt (morado oscuro) | Respuesta (violeta)")
    print(f"  ➜ Guardando en: {output_file}")
    
    # =========================================================================
    # Dibujar el grafo usando graph-tool DIRECTAMENTE
    # =========================================================================
    try:
        graph_draw(
            G,
            pos=pos,
            vertex_text=vertex_text_prop,
            vertex_font_size=12,  # Tamaño de fuente para etiquetas
            vertex_size=vertex_size,
            vertex_fill_color=vertex_fill_color,
            vertex_color=[0, 0, 0, 1],  # Borde negro para nodos
            vertex_pen_width=2,  # Borde moderado
            edge_pen_width=edge_pen_width,
            edge_color=edge_color,
            edge_marker_size=12,  # Tamaño de flecha
            bg_color=[1, 1, 1, 1],  # Fondo blanco
            output_size=(4800, 4800),  # Imagen de alta resolución
            output=str(output_file)
        )
        print(f"✅ Gráfico guardado exitosamente (Resolución: 4800x4800)")
    except Exception as e:
        print(f"❌ Error al generar visualización: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"✅ Gráfico guardado en: {output_file}")
    print(f"  ➜ Question ID: {question_id}")
    print(f"  ➜ Layer: {layer_idx}")
    print(f"  ➜ Nodos mostrados: {G.num_vertices()} / {original_num_nodes}")
    print(f"  ➜ Aristas: {G.num_edges()}")
    print(f"  ➜ Prompt tokens: 0-{prompt_end_idx-1} | Respuesta tokens: {prompt_end_idx}-{num_nodes-1}")


def create_layerwise_comparison(trace, layers_to_show=[0, 15, 31], 
                                output_file='layerwise_comparison.png',
                                attn_threshold=0.01):
    """
    Crea una comparación de grafos entre diferentes capas usando graph-tool con layout sfdp.
    Usa 2 colores HLS (posiciones 6 y 7 de 8) para distinguir prompt vs respuesta.
    Arcos usan colormap viridis.
    """
    num_layers_to_show = len(layers_to_show)
    
    print(f"  ➜ Generando comparación de {num_layers_to_show} capas...")
    
    # Obtener paleta HLS con 8 colores discretos
    hls_palette = sns.color_palette("hls", 8)
    color_prompt = hls_palette[5]    # Morado oscuro (posición 6)
    color_response = hls_palette[6]   # Violeta (posición 7)
    
    # Determinar punto de corte entre prompt y respuesta
    tokens_decoded = trace.get('tokens_decoded', [])
    total_tokens = len(tokens_decoded)
    prompt_end_idx = total_tokens // 2
    
    # Intentar detectar mejor el punto de corte
    for i, token in enumerate(tokens_decoded):
        if token.strip().lower() in ['answer:', 'a:', 'response:', '\n\n']:
            prompt_end_idx = i
            break
    
    # Generar una imagen para cada capa
    temp_files = []
    for idx, layer_idx in enumerate(layers_to_show):
        print(f"    - Procesando capa {layer_idx}...")
        
        # Crear grafo para esta capa
        G, edge_weights, attn_avg = create_attention_graph(trace, layer_idx, attn_threshold)
        num_nodes = G.num_vertices()
        
        # Layout sfdp
        pos = sfdp_layout(G, cooling_step=0.95, epsilon=1e-2, max_iter=500)
        
        # Colores: 2 colores para prompt y respuesta
        vertex_fill_color = G.new_vertex_property("vector<double>")
        for v in G.vertices():
            idx_v = int(v)
            if idx_v < prompt_end_idx:
                r, g, b = color_prompt
            else:
                r, g, b = color_response
            vertex_fill_color[v] = [r, g, b, 0.85]
        
        # Tamaño uniforme de nodos
        vertex_size = G.new_vertex_property("double")
        for v in G.vertices():
            vertex_size[v] = 30
        
        # Propiedades de arcos con viridis
        if edge_weights and len(edge_weights) > 0:
            min_w, max_w = min(edge_weights), max(edge_weights)
            viridis_cmap = cm.get_cmap('viridis')
            
            edge_pen_width = G.new_edge_property("double")
            edge_color = G.new_edge_property("vector<double>")
            
            for e in G.edges():
                weight = G.ep.weight[e]
                norm_weight = (weight - min_w) / (max_w - min_w) if max_w > min_w else 0.5
                edge_pen_width[e] = 1 + norm_weight * 6
                
                rgba = viridis_cmap(norm_weight)
                edge_color[e] = [rgba[0], rgba[1], rgba[2], 0.6 + 0.4 * norm_weight]
        else:
            edge_pen_width = G.new_edge_property("double")
            edge_color = G.new_edge_property("vector<double>")
            for e in G.edges():
                edge_pen_width[e] = 2
                edge_color[e] = [0.5, 0.5, 0.5, 0.3]
        
        # Etiquetas con tokens
        vertex_text = G.vp.label  # Usar los tokens del grafo
        
        # Guardar imagen temporal
        temp_file = str(output_file).replace('.png', f'_temp_layer_{layer_idx}.png')
        temp_files.append(temp_file)
        
        # Dibujar
        graph_draw(
            G,
            pos=pos,
            vertex_text=vertex_text,
            vertex_font_size=10,
            vertex_size=vertex_size,
            vertex_fill_color=vertex_fill_color,
            vertex_color=[0, 0, 0, 1],
            vertex_pen_width=1.5,
            edge_pen_width=edge_pen_width,
            edge_color=edge_color,
            edge_marker_size=8,
            bg_color=[1, 1, 1, 1],
            output_size=(1600, 1600),  # Mayor resolución para cada sub-grafo
            output=temp_file
        )
    
    # Combinar imágenes usando matplotlib
    from PIL import Image
    
    fig, axes = plt.subplots(1, num_layers_to_show, figsize=(8*num_layers_to_show, 8))
    if num_layers_to_show == 1:
        axes = [axes]
    
    for idx, (layer_idx, temp_file) in enumerate(zip(layers_to_show, temp_files)):
        img = Image.open(temp_file)
        axes[idx].imshow(img)
        axes[idx].set_title(f'Layer {layer_idx}', fontsize=16, fontweight='bold')
        axes[idx].axis('off')
        
        # Eliminar archivo temporal
        import os
        os.remove(temp_file)
    
    question_id = trace.get('question_id', 'Unknown')
    fig.suptitle(f'Layerwise Attention Graph Comparison - {question_id}', 
                fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✅ Comparación guardada en: {output_file} (DPI: 600)")
    plt.close()


def create_attention_matrix_heatmap(trace, layer_idx, output_file='attention_heatmap.png'):
    """
    Crea un heatmap de la matriz de atención usando colormap viridis.
    """
    attentions = trace['attentions'][layer_idx]
    attn_avg = attentions.mean(axis=0)  # Promediar heads
    
    tokens_decoded = trace.get('tokens_decoded', [])
    num_tokens = min(len(tokens_decoded), attn_avg.shape[0])
    
    # Recortar si es necesario
    attn_avg = attn_avg[:num_tokens, :num_tokens]
    
    # Crear figura con mayor resolución
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Heatmap con colormap viridis
    im = ax.imshow(attn_avg, cmap='viridis', aspect='auto', interpolation='bilinear')
    
    # Etiquetas de ejes (cada 5 tokens para no saturar)
    step = max(1, num_tokens // 20)
    tick_positions = list(range(0, num_tokens, step))
    tick_labels = [tokens_decoded[i][:10] if i < len(tokens_decoded) else str(i) 
                  for i in tick_positions]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Key Tokens', fontsize=14, fontweight='bold')
    ax.set_ylabel('Query Tokens', fontsize=14, fontweight='bold')
    ax.set_title(f'Attention Matrix - Layer {layer_idx}\n'
                f'Question: {trace.get("question_id", "Unknown")}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Colorbar con etiqueta más visible
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✅ Heatmap guardado en: {output_file} (DPI: 600)")
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
