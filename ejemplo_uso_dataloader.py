"""
Ejemplo de cómo usar el TraceGraphDataset en otro archivo
"""

# Importar la clase del dataloader
from dataloader import TraceGraphDataset
from torch_geometric.loader import DataLoader

# 1. Crear el dataset
dataset = TraceGraphDataset(
    pkl_files_pattern="path/to/your/*.pkl",
    attn_threshold=0.01
)

# 2. Crear el DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0  # Ajustar según tu sistema
)

# 3. Usar en un loop de entrenamiento
for epoch in range(10):
    for batch in loader:
        # batch.x contiene las características de los nodos
        # batch.edge_index contiene los arcos
        # batch.edge_attr contiene los pesos de atención
        
        # Tu código de entrenamiento aquí
        print(f"Batch con {batch.num_graphs} grafos, {batch.num_nodes} nodos totales")
        
        # Acceder a metadatos
        print(f"Question IDs: {batch.question_id}")
        print(f"Layer indices: {batch.layer_idx}")
        
        break  # Solo primer batch como ejemplo
    break  # Solo primera época como ejemplo
