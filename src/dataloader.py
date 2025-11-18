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
    
    OPTIMIZADO PARA MEMORIA: Usa lazy loading, cargando archivos bajo demanda
    y liberando memoria entre accesos.
    
    Si se cargan 100 traces y cada uno tiene 32 capas, este dataset
    reportará un tamaño de 3200 items.
    """
    def __init__(self, pkl_files_pattern, attn_threshold=0.01, lazy_loading=True):
        """
        Inicializa el dataset.
        
        Args:
            pkl_files_pattern (str): Patrón glob para encontrar los archivos .pkl
                                      (e.g., "/path/to/traces_data/*.pkl").
            attn_threshold (float): Umbral para crear arcos de atención. [cite: 194]
            lazy_loading (bool): Si True, carga archivos bajo demanda. Si False, carga todo en memoria (legacy).
        """
        
        self.attn_threshold = attn_threshold
        self.lazy_loading = lazy_loading
        
        print("Buscando archivos pkl/pkl.gz...")
        # Buscar tanto .pkl como .pkl.gz
        file_paths = glob.glob(pkl_files_pattern)
        # Si el patrón no incluye .gz, buscar también archivos comprimidos
        if not pkl_files_pattern.endswith('.gz'):
            file_paths.extend(glob.glob(pkl_files_pattern.replace('.pkl', '.pkl.gz')))
        
        # Filtrar archivos parciales (.part)
        file_paths = [f for f in file_paths if not f.endswith('.part')]
        
        print(f"Encontrados {len(file_paths)} archivos.")
        
        if not file_paths:
            self.num_layers = 0
            self.index_map = []
            self.file_index_map = []
            print("Advertencia: No se encontraron archivos. El dataset está vacío.")
            return
        
        if self.lazy_loading:
            # MODO LAZY: Solo guardar paths y crear índice sin cargar datos
            self.file_paths = sorted(file_paths)
            self.file_index_map = []  # (file_idx, trace_in_file_idx, layer_idx)
            self._file_cache = {}  # Cache LRU simple
            self._cache_max_size = 6  # Aumentado de 2 a 6 para mejor rendimiento
            self._cache_order = []  # Para implementar LRU
            
            # Leer el primer archivo para detectar num_layers y contar traces por archivo
            print("Escaneando estructura de archivos...")
            for file_idx, file_path in enumerate(tqdm(self.file_paths, desc="Indexando")):
                if file_path.endswith('.gz'):
                    with gzip.open(file_path, 'rb') as f:
                        batch_data = pickle.load(f)
                else:
                    with open(file_path, 'rb') as f:
                        batch_data = pickle.load(f)
                
                if file_idx == 0:
                    self.num_layers = len(batch_data[0]['hidden_states'])
                
                # Crear índice para cada trace y capa en este archivo
                for trace_idx in range(len(batch_data)):
                    for layer_idx in range(self.num_layers):
                        self.file_index_map.append((file_idx, trace_idx, layer_idx))
            
            print(f"Detectado {self.num_layers} capas por trace.")
            print(f"Total de grafos a generar (items): {len(self.file_index_map)}")
            total_traces = len(self.file_index_map) // self.num_layers
            print(f" (Equivalente a ~{total_traces} traces * {self.num_layers} capas)")
            
        else:
            # MODO LEGACY: Cargar todo en memoria (NO RECOMENDADO para datasets grandes)
            print("⚠️  ADVERTENCIA: Cargando todo en memoria (lazy_loading=False)")
            print("   Esto puede causar problemas de memoria con datasets grandes.")
            self.all_traces = []
            
            for file_path in tqdm(file_paths, desc="Cargando traces"):
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
        if self.lazy_loading:
            return len(self.file_index_map)
        else:
            return len(self.index_map)
    
    def _load_file(self, file_idx):
        """
        Carga un archivo con cache LRU (Least Recently Used).
        """
        if file_idx in self._file_cache:
            # Mover al final (más recientemente usado)
            self._cache_order.remove(file_idx)
            self._cache_order.append(file_idx)
            return self._file_cache[file_idx]
        
        # Limpiar cache si está lleno
        if len(self._file_cache) >= self._cache_max_size:
            # Eliminar el elemento menos recientemente usado (primero en la lista)
            oldest_key = self._cache_order.pop(0)
            del self._file_cache[oldest_key]
            gc.collect()
        
        # Cargar archivo
        file_path = self.file_paths[file_idx]
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rb') as f:
                batch_data = pickle.load(f)
        else:
            with open(file_path, 'rb') as f:
                batch_data = pickle.load(f)
        
        self._file_cache[file_idx] = batch_data
        self._cache_order.append(file_idx)
        return batch_data
    
    def __getitem__(self, idx):
        """
        Obtiene un único grafo correspondiente a un par (trace, capa).
        """
        if self.lazy_loading:
            if idx >= len(self.file_index_map):
                raise IndexError("Índice fuera de rango")
            
            # 1. Obtener los índices desde el mapa
            file_idx, trace_in_file_idx, layer_idx = self.file_index_map[idx]
            
            # 2. Cargar el archivo (usa cache)
            batch_data = self._load_file(file_idx)
            
            # 3. Obtener el trace específico
            trace = batch_data[trace_in_file_idx]
            
        else:
            # MODO LEGACY
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
        
        OPTIMIZADO PARA MEMORIA: Usa operaciones in-place donde es posible
        y libera memoria intermedia.
        
        Args:
            trace (dict): Diccionario que contiene 'hidden_states' y 'attentions'.
            layer_idx (int): Índice de la capa a convertir en grafo.
        
        Returns:
            Data: Grafo de PyG representando la capa del trace.
        """
        
        # 1. NODOS: hidden_states de la capa
        hidden_states = trace['hidden_states'][layer_idx]  # Shape: (seq_len, hidden_dim)
        attentions = trace['attentions'][layer_idx]        # Shape: (num_heads, seq_len, seq_len)
        
        seq_len, hidden_dim = hidden_states.shape
        
        # Convertir a torch si es necesario (de forma eficiente)
        if isinstance(hidden_states, np.ndarray):
            node_features = torch.from_numpy(hidden_states).float()
        else:
            node_features = hidden_states.float()
        
        num_nodes = node_features.size(0)
        
        # 2. ARCOS: Promedio de atención sobre las cabezas
        if isinstance(attentions, np.ndarray):
            # Convertir y promediar en un solo paso
            attn_avg = torch.from_numpy(attentions).float().mean(dim=0)
        else:
            attn_avg = attentions.float().mean(dim=0)
        
        # Recortar atención a num_nodes (por si hay discrepancia)
        if attn_avg.size(0) > num_nodes:
            attn_avg = attn_avg[:num_nodes, :num_nodes]
        elif attn_avg.size(0) < num_nodes:
            # Pad con zeros si es necesario
            pad_size = num_nodes - attn_avg.size(0)
            attn_avg = F.pad(attn_avg, (0, pad_size, 0, pad_size), value=0.0)
        
        # Aplicar umbral de forma eficiente
        # IMPORTANTE: Respetar causalidad - solo considerar triangular inferior
        # attn_avg[i,j] = cuánta atención el token i presta al token j
        # El arco debe ir: i -> j (el observador apunta a lo observado)
        # Por causalidad en modelos autoregresivos: solo puede haber arcos i->j si j <= i
        
        # Crear máscara causal (triangular inferior)
        causal_mask = torch.tril(torch.ones(num_nodes, num_nodes, dtype=torch.bool))
        
        # Aplicar tanto umbral como máscara causal
        mask = (attn_avg > self.attn_threshold) & causal_mask
        
        # Obtener edge_index de forma eficiente
        indices = mask.nonzero(as_tuple=False).t()
        
        if indices.numel() > 0:
            # Validar índices
            valid_mask = (indices[0] < num_nodes) & (indices[1] < num_nodes)
            indices = indices[:, valid_mask]
            
            if indices.numel() > 0:
                # Crear edge_index: i -> j donde attn_avg[i,j] > threshold
                # indices[0] = i (observador), indices[1] = j (observado)
                edge_index = torch.stack([indices[0], indices[1]], dim=0)
                
                # Obtener edge_attr
                edge_attr = attn_avg[mask][valid_mask].unsqueeze(1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        # 3. Crear el objeto Data de PyG
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        # 4. Añadir metadatos adicionales al grafo
        data.question_id = trace.get('question_id', 'unknown')
        data.answer = trace.get('generated_answer_clean', '')
        data.tokens_decoded = trace.get('tokens_decoded', [])
        data.num_nodes = num_nodes
        data.layer_idx = layer_idx
        
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
