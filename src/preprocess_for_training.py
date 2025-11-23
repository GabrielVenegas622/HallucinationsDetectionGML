import pickle
import numpy as np
from sklearn.decomposition import IncrementalPCA
from pathlib import Path
import torch
from torch_geometric.data import Data
import random
from tqdm import tqdm
import argparse
import gc
import gzip

# --- Configuration ---
PCA_COMPONENTS = 64
SHARD_SIZE = 2000
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

def fit_reducer_in_streaming(all_files, num_files_str):
    """
    Fits an IncrementalPCA model by streaming through files one by one
    to avoid loading all data into memory, thus preventing OOM errors.
    """
    print("--- Phase 1: Calibrating PCA in Streaming Mode ---")
    reducer = IncrementalPCA(n_components=PCA_COMPONENTS)

    if num_files_str.lower() == 'all':
        files_to_process = all_files
        print(f"Using all {len(all_files)} files to fit PCA.")
    else:
        try:
            num_files = int(num_files_str)
            if num_files <= 0:
                raise ValueError("Number of files must be positive.")
            if num_files > len(all_files):
                print(f"Warning: Requested {num_files} files, but only {len(all_files)} available. Using all of them.")
                num_files = len(all_files)
            files_to_process = random.sample(all_files, k=num_files)
            print(f"Randomly sampling {num_files} files to fit PCA.")
        except ValueError:
            raise ValueError(f"Invalid value for --pca-fit-files: '{num_files_str}'. Must be an integer or 'all'.")

    for pkl_file in tqdm(files_to_process, desc="Streaming PCA fit"):
        try:
            if pkl_file.suffix == '.gz':
                with gzip.open(pkl_file, 'rb') as f: data = pickle.load(f)
            else:
                with open(pkl_file, 'rb') as f: data = pickle.load(f)

            embeddings_chunk = []
            for trace_dict in data:
                for layer_embedding in trace_dict['hidden_states']:
                    if layer_embedding is not None and layer_embedding.shape[0] > 0:
                        embeddings_chunk.append(layer_embedding)
            
            if not embeddings_chunk:
                continue

            chunk_matrix = np.concatenate(embeddings_chunk, axis=0)

            # Critical Safety Check: n_samples must be >= n_components
            if chunk_matrix.shape[0] > PCA_COMPONENTS:
                reducer.partial_fit(chunk_matrix)
            
            del data, embeddings_chunk, chunk_matrix
            gc.collect()

        except Exception as e:
            tqdm.write(f"\nWarning: Failed to process file {pkl_file} during PCA fit: {e}")

    print("PCA calibration complete.")
    return reducer

def process_trace(trace, reducer):
    """
    Processes a single raw trace into a sequence of graph Data objects.
    - Applies PCA to reduce embedding dimensions.
    - Builds a sparse, causal attention graph where each token attends to a
      subset of previous tokens.
    """
    processed_layers = []
    for layer_embedding, layer_attention_avg in trace:
        if layer_embedding is None or layer_embedding.shape[0] == 0:
            x = torch.empty((0, PCA_COMPONENTS), dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            num_nodes = layer_embedding.shape[0]
            
            # 1. Apply PCA transformation
            x = reducer.transform(layer_embedding).astype(np.float32)
            x = torch.from_numpy(x)
            
            # 2. Build sparse, causal graph from attention
            source_nodes, target_nodes, edge_weights = [], [], []
            adj = layer_attention_avg
            
            for i in range(1, num_nodes):  # Start from 1, as token 0 has no predecessors
                # Attention from current token `i` to all previous tokens `j < i`
                attentions_to_previous = adj[i, :i]
                
                # Dynamic Top-K based on number of previous tokens
                k = max(1, int(i * TOP_K_PERCENTILE))
                
                # Find indices of the top-k attentions
                if len(attentions_to_previous) > k:
                    # Use argpartition for efficiency, it finds the k-largest items
                    # without a full sort.
                    top_k_source_indices = np.argpartition(attentions_to_previous, -k)[-k:]
                else:
                    top_k_source_indices = np.arange(i)  # Take all predecessors

                # Add edges: source (j) -> target (i)
                source_nodes.extend(top_k_source_indices)
                target_nodes.extend([i] * len(top_k_source_indices))
                edge_weights.extend(attentions_to_previous[top_k_source_indices])

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
    
    graphs = [item['graphs'] for item in shard_data]
    question_ids = [item['question_id'] for item in shard_data]
    
    save_path = Path(output_dir) / f'sharded_data_part_{shard_num}.pt'
    
    # Note: Labels are not part of this preprocessing step.
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

    # --- Phase 1: Streaming PCA Calibration ---
    reducer = fit_reducer_in_streaming(input_files, args.pca_fit_files)

    # --- Phase 2: Processing and Sharding ---
    print("\n--- Phase 2: Processing and Sharding Data ---")
    
    accumulator = []
    shard_counter = 0

    for pkl_file in tqdm(input_files, desc="Processing files into shards"):
        try:
            if pkl_file.suffix == '.gz':
                with gzip.open(pkl_file, 'rb') as f: data = pickle.load(f)
            else:
                with open(pkl_file, 'rb') as f: data = pickle.load(f)
            
            for trace_dict in data:
                question_id = trace_dict['question_id']
                
                # Average attention heads and create the trace format for processing
                raw_attentions = [attn.mean(axis=0) for attn in trace_dict['attentions']]
                raw_trace = list(zip(trace_dict['hidden_states'], raw_attentions))

                processed_graphs = process_trace(raw_trace, reducer)
                
                accumulator.append({
                    'graphs': processed_graphs,
                    'question_id': question_id
                })
                
                if len(accumulator) >= SHARD_SIZE:
                    save_shard(accumulator, args.output_dir, shard_counter)
                    accumulator.clear()
                    gc.collect()
                    shard_counter += 1
        except Exception as e:
            tqdm.write(f"\nWarning: Failed to process file {pkl_file} during sharding: {e}")
    
    if accumulator:
        save_shard(accumulator, args.output_dir, shard_counter)

    print("\nPreprocessing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Optimized script to preprocess trace data with streaming PCA calibration 
        and data sharding to handle very large datasets without OOM errors.""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing raw .pkl or .pkl.gz trace files.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the processed .pt shard files.')
    parser.add_argument(
        '--pca-fit-files',
        type=str,
        default='50',
        help="Number of files to stream for fitting PCA. Use 'all' to use all files. (default: 50)"
    )
    
    args = parser.parse_args()
    main(args)