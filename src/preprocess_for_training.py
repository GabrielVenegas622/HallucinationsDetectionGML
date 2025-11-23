import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import torch
from torch_geometric.data import Data
import random
from tqdm import tqdm
import argparse
import sys
import gc
import gzip

# --- Configuration ---
SVD_COMPONENTS = 64
SHARD_SIZE = 2000  # Number of traces per shard file
SVD_FIT_MEM_LIMIT_GB = 9
TOP_K_PERCENTILE = 0.35

def get_all_pkl_files(input_dir):
    """Finds all .pkl and .pkl.gz files in the input directory."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    files = list(input_path.glob('*.pkl')) + list(input_path.glob('*.pkl.gz'))
    if not files:
        raise FileNotFoundError(f"No .pkl or .pkl.gz files found in {input_dir}")
    print(f"Found {len(files)} .pkl/.pkl.gz files to process.")
    return files

def fit_svd_on_sample(files, memory_limit_gb):
    """
    Samples embeddings from .pkl files up to a memory limit, then fits a
    TruncatedSVD model on them.
    """
    print(f"--- Phase 1: Calibrating SVD (Memory Limit: ~{memory_limit_gb}GB) ---")
    memory_limit_bytes = memory_limit_gb * 1024**3
    
    all_embeddings = []
    current_size = 0
    files_sampled_count = 0

    # Shuffle files to get a more representative sample if we hit the memory limit early
    random.shuffle(files)

    for pkl_file in tqdm(files, desc="Sampling embeddings for SVD"):
        try:
            if pkl_file.suffix == '.gz':
                with gzip.open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
            files_sampled_count += 1
            
            for trace_dict in data:
                for layer_embedding in trace_dict['hidden_states']:
                    if layer_embedding is not None and layer_embedding.shape[0] > 0:
                        all_embeddings.append(layer_embedding)
                        current_size += layer_embedding.nbytes
                
                if current_size >= memory_limit_bytes:
                    break
            if current_size >= memory_limit_bytes:
                break
        except Exception as e:
            print(f"Warning: Could not read or process {pkl_file}: {e}")
            
    if not all_embeddings:
        raise ValueError("Could not sample any embeddings. Check input files.")

    print(f"Sampled from {files_sampled_count} files.")
    print(f"Collected {len(all_embeddings)} embedding matrices, total size {current_size / 1024**3:.2f} GB.")
    
    concatenated_embeddings = np.concatenate(all_embeddings, axis=0)
    del all_embeddings; gc.collect()

    print(f"Fitting TruncatedSVD on matrix of shape {concatenated_embeddings.shape}")
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
    svd.fit(concatenated_embeddings)
    
    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"SVD FITTED. Total explained variance by {SVD_COMPONENTS} components: {explained_variance:.4f}")
    
    return svd

def process_trace(trace, svd_transformer):
    """
    Processes a single raw trace:
    1. Applies SVD to node embeddings.
    2. Prunes the graph based on attention scores (dynamic top-k).
    3. Converts each layer to a PyG Data object.
    """
    processed_layers = []
    for layer_embedding, layer_attention in trace:
        if layer_embedding is None or layer_embedding.shape[0] == 0:
            x = torch.empty((0, SVD_COMPONENTS), dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            num_nodes = layer_embedding.shape[0]
            
            # Apply SVD transformation
            x = svd_transformer.transform(layer_embedding)
            x = torch.tensor(x, dtype=torch.float32)
            
            # Prune edges dynamically based on attention
            source_nodes, target_nodes, edge_weights = [], [], []
            adj = layer_attention # Assuming dense numpy array

            for i in range(num_nodes):
                outgoing_attentions = adj[i, :]
                # Find potential neighbors (non-zero attention)
                neighbor_indices = np.where(outgoing_attentions > 0)[0]
                
                if len(neighbor_indices) > 0:
                    # Dynamic Top-K for pruning
                    k = max(1, int(len(neighbor_indices) * TOP_K_PERCENTILE))
                    
                    # Efficiently find top-k indices
                    top_k_local_indices = np.argpartition(outgoing_attentions[neighbor_indices], -k)[-k:]
                    top_k_neighbor_indices = neighbor_indices[top_k_local_indices]
                    
                    source_nodes.extend([i] * len(top_k_neighbor_indices))
                    target_nodes.extend(top_k_neighbor_indices)
                    edge_weights.extend(outgoing_attentions[top_k_neighbor_indices])

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        processed_layers.append(graph_data)
        
    return processed_layers

def save_shard(shard_data, output_dir, shard_num):
    """Shuffles and saves a list of processed traces to a .pt file."""
    if not shard_data:
        return
        
    print(f"\nSaving shard {shard_num} with {len(shard_data)} traces...")
    
    random.shuffle(shard_data)
    
    # Unzip from list of dicts to dict of lists
    graphs = [item['graphs'] for item in shard_data]
    question_ids = [item['question_id'] for item in shard_data]
    
    save_path = Path(output_dir) / f'sharded_data_part_{shard_num}.pt'
    
    # Labels are not processed in this script. They should be added in a later step.
    torch.save({
        'graphs': graphs,
        'question_ids': question_ids
    }, save_path)
    print(f"Shard {shard_num} saved to {save_path}")

def main(args):
    """Main script execution."""
    input_files = get_all_pkl_files(args.input_dir)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: SVD Calibration ---
    svd_transformer = fit_svd_on_sample(input_files, SVD_FIT_MEM_LIMIT_GB)

    # --- Phase 2: Processing and Sharding ---
    print("\n--- Phase 2: Processing and Sharding Data ---")
    
    accumulator = []
    shard_counter = 0

    progress_bar = tqdm(total=len(input_files), desc="Processing files")
    for pkl_file in input_files:
        try:
            if pkl_file.suffix == '.gz':
                with gzip.open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
            
            # The loaded data is a list of trace dictionaries
            for trace_dict in data:
                question_id = trace_dict['question_id']
                
                # Average attention heads to get a 2D matrix per layer
                raw_attentions = [attn.mean(axis=0) for attn in trace_dict['attentions']]
                # Create the trace format expected by process_trace
                raw_trace = list(zip(trace_dict['hidden_states'], raw_attentions))

                processed_graphs = process_trace(raw_trace, svd_transformer)
                
                accumulator.append({
                    'graphs': processed_graphs,
                    'question_id': question_id
                    # Note: Labels are not included as they are not in the source .pkl files.
                    # They should be added in a separate step if needed.
                })
                
                # Check if the accumulator is full
                if len(accumulator) >= SHARD_SIZE:
                    save_shard(accumulator, args.output_dir, shard_counter)
                    accumulator.clear()
                    gc.collect()
                    shard_counter += 1
        except Exception as e:
            print(f"\nWarning: Failed to process file {pkl_file}: {e}")
        progress_bar.update(1)
    
    progress_bar.close()

    # Save any remaining data in the last shard
    if accumulator:
        save_shard(accumulator, args.output_dir, shard_counter)

    print("\nPreprocessing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess trace data with SVD and sharding.")
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing raw .pkl trace files.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the processed .pt shard files.')
    
    args = parser.parse_args()
    main(args)
