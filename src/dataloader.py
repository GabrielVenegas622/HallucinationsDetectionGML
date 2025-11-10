import glob
import pickle
import gzip
import torch
import gc  
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm


class TraceGraphDataset(TorchDataset):
    """
    Un Dataset de PyTorch que carga las trazas .pkl y las convierte
    en un grafo de PyG POR CADA CAPA de cada trace.
    
    Si se cargan 100 traces y cada uno tiene 32 capas, este dataset
    reportará un tamaño de 3200 items.
    """
    def __init__(self, pkl_files_pattern, attn_threshold=0.01):
        """
        Inicializa el dataset.
        
        Args:
            pkl_files_pattern (str): Patrón glob para encontrar los archivos .pkl
                                      (e.g., "/path/to/traces_data/*.pkl").
            attn_threshold (float): Umbral para crear arcos de atención. [cite: 194]
        """
        
        self.attn_threshold = attn_threshold
        self.all_traces = []
        
        print("Buscando archivos pkl/pkl.gz...")
        # Buscar tanto .pkl como .pkl.gz
        file_paths = glob.glob(pkl_files_pattern)
        # Si el patrón no incluye .gz, buscar también archivos comprimidos
        if not pkl_files_pattern.endswith('.gz'):
            file_paths.extend(glob.glob(pkl_files_pattern.replace('.pkl', '.pkl.gz')))
        
        print(f"Encontrados {len(file_paths)} archivos.")
        
        # Cargar los traces en memoria RAM
        for file_path in tqdm(file_paths, desc="Cargando traces"):
            # Detectar si el archivo está comprimido
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rb') as f:
                    batch_data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    batch_data = pickle.load(f)
            
            self.all_traces.extend(batch_data)
                
        print("Carga de traces en memoria RAM completada.")
        
        if not self.all_traces:
            self.num_layers = 0
            self.index_map = []
            print("Advertencia: No se cargaron traces. El dataset está vacío.")
            return
        
        self.num_layers = len(self.all_traces[0]['hidden_states'])
        print(f"Detectado {self.num_layers} capas por trace (ej. Llama-2-7B tiene 32).") 
        
        
        # Crear el mapa de índices (trace_idx, layer_idx)
        # Esto "aplana" la estructura
        self.index_map = []
        num_traces = len(self.all_traces)
        for trace_idx in range(num_traces):
            for layer_idx in range(self.num_layers):
                self.index_map.append((trace_idx, layer_idx))
        
        print(f"Total de grafos a generar (items): {len(self.index_map)}")
        print(f" (Equivalente a {num_traces} traces * {self.num_layers} capas)")
        gc.collect()


    def __len__(self):
        """
        Retorna el número total de grafos en el dataset. (total_traces * num_layers).
        """
        return len(self.index_map)
    
    def __getitem__(self, idx):
        """
        Obtiene un único grafo correspondiente a un par (trace, capa).
        """
        if idx >= len(self.index_map):
            raise IndexError("Índice fuera de rango")
            
        # 1. Obtener los índices de trace y capa desde el mapa
        trace_idx, layer_idx = self.index_map[idx]
        
        # 2. Obtener el diccionario del trace
        trace = self.all_traces[trace_idx]
        
        # 3. Convertir ese trace y esa capa específica a un grafo
        return self._trace_to_graph(trace, layer_idx)
    
    def _trace_to_graph(self, trace, layer_idx):
        """
        Convierte un trace y una capa específica en un grafo de PyG.
        
        Args:
            trace (dict): Diccionario que contiene 'hidden_states' y 'attentions'.
            layer_idx (int): Índice de la capa a convertir en grafo.
        
        Returns:
            Data: Grafo de PyG representando la capa del trace.
        """
        
        
        # hidden_states = trace['hidden_states'][layer_idx]  # Shape: (seq_len, hidden_dim)
        # attentions = trace['attentions'][layer_idx]        # Shape: (num_heads, seq_len, seq_len)
        
        # seq_len, hidden_dim = hidden_states.shape
        # num_heads = attentions.shape[0]
        
        # # 1. Crear nodos con características de hidden states
        # x = torch.tensor(hidden_states, dtype=torch.float)  # Shape: (seq_len, hidden_dim)
        
        # # 2. Crear arcos basados en atención
        # edge_index_list = []
        # edge_attr_list = []
        
        # for head in range(num_heads):
        #     attn_matrix = attentions[head]  # Shape: (seq_len, seq_len)
        #     src, dst = np.where(attn_matrix > self.attn_threshold)
        #     edge_index_list.append(torch.tensor([src, dst], dtype=torch.long))
        #     edge_attr_list.append(torch.tensor(attn_matrix[src, dst], dtype=torch.float).unsqueeze(1))
        
        # if edge_index_list:
        #     edge_index = torch.cat(edge_index_list, dim=1)  # Shape: (2, num_edges)
        #     edge_attr = torch.cat(edge_attr_list, dim=0)    # Shape: (num_edges, 1)
        # else:
        #     edge_index = torch.empty((2, 0), dtype=torch.long)
        #     edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        # # 3. Crear el objeto Data de PyG
        # graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # return graph_data
        
        
        node_features = torch.tensor(
            trace['hidden_states'][layer_idx], # [cite: 69]
            dtype=torch.float
        )
        
        # Obtener número de nodos (tokens en la secuencia)
        # Esto DEBERÍA incluir prompt + respuesta si se extrajo correctamente
        num_nodes = node_features.shape[0]
        
        # 2. Arcos (Edges) y Atributos de Arco (Edge Attributes)
        # Extrae las atenciones de la capa específica
        attentions_layer = torch.tensor(
            trace['attentions'][layer_idx], # [cite: 70]
            dtype=torch.float
        )
        
        # Promediamos las cabezas de atención [cite: 191]
        attn_avg = attentions_layer.mean(axis=0)
        
        # VALIDACIÓN CRÍTICA: Verificar consistencia entre hidden_states y attentions
        # IMPORTANTE: Si hidden_states es del prompt+respuesta completo, attn_avg DEBE ser cuadrada del mismo tamaño
        # Si hay mismatch, puede indicar:
        #   1. Las atenciones fueron extraídas con padding fijo (ej: 512x512)
        #   2. Solo se guardaron hidden_states de la respuesta, no del prompt
        #   3. Hay un bug en la extracción de traces
        
        if attn_avg.shape[0] != num_nodes or attn_avg.shape[1] != num_nodes:
            # WARNING: Dimensiones no coinciden
            # El dataloader corregirá esto recortando o rellenando
            
            # Caso 1: attn_avg es MÁS GRANDE que num_nodes (padding o secuencia completa vs parcial)
            if attn_avg.shape[0] > num_nodes or attn_avg.shape[1] > num_nodes:
                # Recortar a los nodos reales
                attn_avg = attn_avg[:num_nodes, :num_nodes]
                
                # WARNING si la diferencia es muy grande (indica posible problema en extracción)
                if attn_avg.shape[0] > num_nodes * 2:
                    print(f"⚠️  WARNING: Trace {trace.get('question_id', '?')}, capa {layer_idx}: "
                          f"Atenciones ({attentions_layer.shape[1]}x{attentions_layer.shape[2]}) >> "
                          f"hidden_states ({num_nodes} nodos). Recortando automáticamente.")
                    if num_nodes == 1:
                        print(f"   ⚠️  CRÍTICO: Solo 1 nodo pero atenciones grandes. "
                              f"¿Se guardó solo la respuesta y no el prompt?")
            
            # Caso 2: attn_avg es MÁS PEQUEÑO que num_nodes (caso raro)
            elif attn_avg.shape[0] < num_nodes or attn_avg.shape[1] < num_nodes:
                # Rellenar si es más pequeño (caso muy raro)
                pad_size = num_nodes
                padded = torch.zeros((pad_size, pad_size), dtype=attn_avg.dtype)
                padded[:attn_avg.shape[0], :attn_avg.shape[1]] = attn_avg
                attn_avg = padded
                print(f"⚠️  WARNING: Trace {trace.get('question_id', '?')}, capa {layer_idx}: "
                      f"Atenciones ({attentions_layer.shape[1]}x{attentions_layer.shape[2]}) < "
                      f"hidden_states ({num_nodes} nodos). Rellenando con zeros.")
        
        # Aplicamos el umbral [cite: 194]
        mask = attn_avg > self.attn_threshold
        
        # PyG espera edge_index en formato [2, num_edges]
        # (Fuente -> Destino)
        indices = mask.nonzero(as_tuple=False).t() # [2, num_edges]
        
        # Verificar que los índices estén en rango válido
        # (No deberían estar fuera de rango después del recorte, pero validamos por seguridad)
        if indices.numel() > 0:
            # Filtrar índices fuera de rango 
            valid_mask = (indices[0] < num_nodes) & (indices[1] < num_nodes)
            
            if not valid_mask.all():
                num_invalid = (~valid_mask).sum().item()
                print(f"⚠️  WARNING: {num_invalid} arcos con índices fuera de rango [0, {num_nodes}) "
                      f"en trace {trace.get('question_id', '?')}, capa {layer_idx}. Filtrando.")
            
            indices = indices[:, valid_mask]
            
            # Según[cite: 195], el flujo es (j -> i), por lo que j=fuente, i=destino
            # indices[0] es 'i' (fila, destino)
            # indices[1] es 'j' (columna, fuente)
            if indices.numel() > 0:
                edge_index = torch.stack([indices[1], indices[0]], dim=0)
                
                # Obtener edge_attr correspondiente
                # Necesitamos reconstruir la máscara válida original
                edge_attr_values = attn_avg[mask]
                if valid_mask.numel() > 0:
                    edge_attr = edge_attr_values[valid_mask]
                else:
                    edge_attr = edge_attr_values
                
                # Asegurar que edge_attr sea 2D
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(1)
            else:
                # Todos los índices fueron inválidos
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            # No hay arcos que superen el threshold
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        # 3. Crear el objeto Data de PyG
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        # 4. Añadir metadatos adicionales al grafo
        data.question_id = trace.get('question_id', 'unknown') # [cite: 65]
        data.answer = trace.get('generated_answer_clean', '') # [cite: 66]
        data.tokens_decoded = trace.get('tokens_decoded', []) # [cite: 72]
        data.num_nodes = num_nodes
        
        # --- NUEVO METADATO ---
        data.layer_idx = layer_idx # Guardamos de qué capa vino
        
        return data

if __name__ == "__main__":
    
    # 1. Definir la ruta a tus archivos .pkl
    # CAMBIA ESTO a la ruta correcta donde guardaste tus batches
    DATA_PATH_PATTERN = "*.pkl" 
    
    # 2. Crear la instancia del Dataset
    print("Creando el TraceGraphDataset (modo 1 grafo por capa)...")
    graph_dataset = TraceGraphDataset(
        pkl_files_pattern=DATA_PATH_PATTERN,
        attn_threshold=0.01
    )
    
    if len(graph_dataset) > 0:
        # 3. Crear el DataLoader de PyTorch Geometric
        batch_size = 16 # Puedes ajustar esto
        
        print(f"Creando el DataLoader de PyG con batch_size={batch_size}...")
        graph_loader = DataLoader(
            graph_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )

        # 4. Iterar sobre el DataLoader
        print("Iterando sobre el primer batch del DataLoader...")
        print("Este batch contendrá una mezcla de grafos de diferentes traces y capas.")
        
        for i, batch in enumerate(graph_loader):
            print("\n--- ¡Batch 0 cargado! ---")
            
            print(f"Tipo de objeto: {type(batch)}")
            
            # 'batch.batch' es un vector que mapea cada nodo a su grafo original
            # (en este caso, un grafo (trace, capa))
            print(f"Nodos en el batch: {batch.num_nodes}")
            print(f"Arcos en el batch: {batch.num_edges}")
            print(f"Grafos en el batch: {batch.num_graphs} (debería ser {batch_size})")
            
            # NUEVO: Verificamos las capas y IDs en el batch
            print(f"Capas en este batch (batch.layer_idx): {batch.layer_idx}")
            print(f"IDs de preguntas (batch.question_id): {batch.question_id}")
            
            break
            
        print("\n--- Verificación del tamaño del Dataset ---")
        print(f"Total de items en el dataset (grafos): {len(graph_dataset)}")
        print(f" (Ej. si tenías 5000 traces [cite: 144] y 32 capas[cite: 129],")
        print(f"  el total debería ser 5000 * 32 = 160,000)")
    else:
        print(f"No se encontraron traces. Verifica el patrón de ruta: {DATA_PATH_PATTERN}")
