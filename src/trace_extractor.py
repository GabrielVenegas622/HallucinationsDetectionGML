import torch
import random
import os
import numpy as np
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from datasets import load_dataset


def seed_everything(seed: int):
    """
    Fija todas las semillas de generadores de números aleatorios para garantizar
    la replicabilidad en la generación por sampling (muestreo).
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED_VALUE = 41

def extract_activations_and_attentions(model, tokenizer, question, max_new_tokens=64):
    """
    Extrae las activaciones (hidden states) y atenciones del estado final
    de la generación (después de generar todos los tokens).
    
    Este método realiza una "foto final" del estado completo del modelo después
    de generar toda la respuesta. Captura activaciones y atenciones para TODA
    la secuencia (prompt + respuesta generada), lo cual es esencial para:
    
    1. Analizar las atenciones que la respuesta presta al prompt (contexto)
    2. Construir un grafo completo de dependencias entre todos los tokens
    3. Detectar patrones de alucinación considerando la interacción prompt-respuesta
    
    IMPORTANTE: Solo se realiza UN forward pass al final, capturando el estado
    completo de toda la secuencia. No se guardan estados intermedios de cada
    paso de generación.
    
    Args:
        model: Modelo de lenguaje cargado
        tokenizer: Tokenizer correspondiente al modelo
        question: Pregunta a responder
        max_new_tokens: Número máximo de tokens a generar
        
    Returns:
        dict con:
            - 'generated_answer_clean': respuesta completa generada (str)
            - 'hidden_states': lista de arrays por capa, cada uno con shape 
                               [seq_len_total, hidden_dim], donde seq_len_total 
                               incluye tokens de prompt + respuesta
            - 'attentions': lista de arrays por capa, cada uno con shape 
                           [num_heads, seq_len_total, seq_len_total], matriz completa
                           de atención que incluye interacciones prompt↔respuesta
            - 'tokens': IDs de TODOS los tokens (prompt + respuesta), array completo
            - 'tokens_decoded': lista de strings, cada token decodificado individualmente
                               (prompt + respuesta), útil para visualización en grafos
    """
    # Preparar el prompt (compatible con Llama)
    prompt_text = f"Answer the question concisely in one sentence.\n\nQuestion: {question}\nAnswer:"
    prompt_text = f"Answer the question concisely. Q: {question} A:"
    
    prompt = tokenizer(prompt_text, return_tensors='pt').to(model.device)
    
    # Asegurar que el tokenizer tiene tokens especiales configurados
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    
    # Generar con activaciones y atenciones
    with torch.no_grad():
        generation_output = model.generate(
            **prompt,
            num_beams=5,
            num_return_sequences=1,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_attentions=True,
            output_hidden_states=True
        )
    
    # Decodificar la respuesta completa (sin prompt, solo para guardar el texto)
    prompt_length = prompt["input_ids"].shape[-1]
    generated_answer_clean = tokenizer.decode(
        generation_output.sequences[0, prompt_length:],
        skip_special_tokens=True
    )
    
    # Número de tokens realmente generados (sin prompt)
    num_tokens_generated = generation_output.sequences.shape[-1] - prompt_length
    
    # Extraer hidden states y attentions del ESTADO FINAL (último paso de generación)
    num_generated_steps = len(generation_output.hidden_states)
    num_layers = len(generation_output.hidden_states[0]) - 1  # -1 porque incluye embeddings
    
    # Índice del último paso de generación
    final_step_idx = num_generated_steps - 1
    
    # Longitud total de la secuencia (prompt + todos los tokens generados)
    seq_len_total = prompt_length + num_tokens_generated
    
    # ========================================================================
    # ESTADO FINAL: Foto completa de activaciones y atenciones
    # ========================================================================
    
    # Organizar hidden states: [capa] -> array[seq_len_total, hidden_dim]
    # Extraemos SOLO el último paso que contiene el estado completo de toda la secuencia
    hidden_states_by_layer = []
    for layer_idx in range(1, num_layers + 1):  # Empezar desde 1 para saltar embeddings
        # Estado final de esta capa (incluye prompt + todos los tokens generados)
        final_state = generation_output.hidden_states[final_step_idx][layer_idx]
        # Shape: [batch=1, seq_len, hidden_dim] -> [seq_len_total, hidden_dim]
        final_state_full = final_state[0, :seq_len_total, :].cpu().numpy()
        hidden_states_by_layer.append(final_state_full)
    
    # Organizar attentions: [capa] -> array[num_heads, seq_len_total, seq_len_total]
    # Extraemos SOLO el último paso que contiene la matriz de atención completa
    attentions_by_layer = []
    for layer_idx in range(num_layers):
        # Atenciones finales de esta capa
        final_attn = generation_output.attentions[final_step_idx][layer_idx]
        # Shape: [batch=1, num_heads, seq_len, seq_len] -> [num_heads, seq_len_total, seq_len_total]
        final_attn_full = final_attn[0, :, :seq_len_total, :seq_len_total].cpu().numpy()
        attentions_by_layer.append(final_attn_full)
      
    # Tokens completos (IDs): prompt + respuesta generada
    all_tokens = generation_output.sequences[0, :seq_len_total].cpu().numpy()
    
    # Decodificar cada token individualmente para visualización en grafos
    # Esto permite que cada nodo del grafo tenga una etiqueta legible
    tokens_decoded = []
    for token_id in all_tokens:
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        tokens_decoded.append(token_text)
    
    return {
        'hidden_states': hidden_states_by_layer,  # [num_layers] cada uno: [seq_len_total, hidden_dim]
        'attentions': attentions_by_layer,  # [num_layers] cada uno: [num_heads, seq_len_total, seq_len_total]
        'tokens': all_tokens,  # IDs de TODOS los tokens (prompt + respuesta)
        'tokens_decoded': tokens_decoded,  # Strings de TODOS los tokens decodificados
        'generated_answer_clean': generated_answer_clean  # Respuesta limpia (solo texto generado)
    }

HF_NAMES = {
    'qwen_2.5_6B' : '__',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
}


def main(args):
    # Extraer nombre base del modelo (sin organización)
    model_name = args.model_id
    model_load = HF_NAMES[f'{args.model_id}']
    
    # Nombre del dataset (triviaqa o truthfulqa)
    dataset_name = args.dataset.lower()
    
    # Configuración de batches para gestión de memoria
    BATCH_SIZE = 1_000
    
    print(f"Cargando modelo: {model_load}")
    tokenizer = AutoTokenizer.from_pretrained(model_load)
    
    # Detectar si el modelo ya está cuantizado (bnb-4bit o bnb-8bit en el nombre)
    is_prequantized = "bnb-4bit" in args.model_id.lower() or "bnb-8bit" in args.model_id.lower()
    
    if is_prequantized:
        print("✅ Modelo pre-cuantizado detectado, cargando directamente...")
        # Para modelos pre-cuantizados de Unsloth, NO usar quantization_config
        model = AutoModelForCausalLM.from_pretrained(
            model_load,
            attn_implementation="eager",  # Necesario para extraer atenciones
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print("⚙️  Aplicando cuantización de 4-bit con BitsAndBytes...")
        # Configuración óptima de cuantización 4-bit para RTX 4060 8GB
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Cuantización a 4 bits
            bnb_4bit_compute_dtype=torch.float16, # Tipo de dato para cálculos
            bnb_4bit_quant_type="nf4"             # Tipo de cuantización: NormalFloat4 (óptimo)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_load,
            attn_implementation="eager",  # Necesario para extraer atenciones
            quantization_config=bnb_config,
            device_map="auto",            # Distribución automática en GPU
            dtype=torch.float16     # Tipo de dato base
        )
    
    num_layers = len(model.model.layers)
    print(f"Número de capas del modelo: {num_layers}")
    
    # Cargar dataset según especificación
    print(f"\nCargando dataset {dataset_name}...")
    if args.dataset.lower() == 'triviaqa':
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        id_field = 'question_id'
        question_field = 'question'
    elif args.dataset.lower() == 'truthfulqa':
        dataset = load_dataset("truthful_qa", "generation", split="validation")
        id_field = None  # TruthfulQA no tiene ID único, usaremos índice
        question_field = 'question'
    else:
        raise ValueError(f"Dataset no soportado: {args.dataset}. Use 'triviaqa' o 'truthfulqa'")
    
    # Limitar el número de ejemplos si se especifica
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    
    total_examples = len(dataset)
    print(f"Número de muestras a procesar: {total_examples}")
    print(f"Tamaño de batch: {BATCH_SIZE} traces por archivo")
    print(f"Archivos esperados: {(total_examples + BATCH_SIZE - 1) // BATCH_SIZE}")
    
    # Crear directorio para guardar los traces
    output_dir = Path("./traces_data")
    output_dir.mkdir(exist_ok=True)
    
    # Variables para el procesamiento por batches
    current_batch = []
    batch_number = 0
    total_processed = 0
    total_errors = 0
    
    # Procesar cada ejemplo del dataset
    for idx, example in enumerate(tqdm(dataset, desc="Extrayendo trazas")):
        question = example[question_field]
        
        # Obtener ID único según el dataset
        if id_field and id_field in example:
            unique_id = example[id_field]
        else:
            unique_id = f"{dataset_name}_{idx}"
        
        try:
            # Extraer activaciones y atenciones
            traces = extract_activations_and_attentions(
                model=model,
                tokenizer=tokenizer,
                question=question,
                max_new_tokens=64,
            )
            
            # Añadir solo el ID como metadata
            traces['question_id'] = unique_id
            
            current_batch.append(traces)
            total_processed += 1
            
            # Mostrar ejemplo cada 10 muestras
            if idx % 10 == 0:
                print(f"\n--- Ejemplo {idx} (Batch actual: {len(current_batch)}/{BATCH_SIZE}) ---")
                print(f"Question ID: {unique_id}")
                print(f"Pregunta: {question}")
                print(f"Respuesta limpia: {traces['generated_answer_clean']}")
                print(f"Total de tokens (prompt + respuesta): {len(traces['tokens'])}")
                print(f"Primeros 5 tokens decodificados: {traces['tokens_decoded'][:5]}...")
                print(f"Últimos 5 tokens decodificados: {traces['tokens_decoded'][-5:]}...")
            
        except Exception as e:
            print(f"\n⚠️  Error procesando ejemplo {idx}: {e}")
            total_errors += 1
            continue
        
        # Guardar batch cuando alcance el tamaño especificado
        if len(current_batch) >= BATCH_SIZE:
            # Formato: <modelo>_<dataset>_<num_batch>
            output_file = output_dir / f"{model_name}_{dataset_name}_batch_{batch_number:04d}.pkl"
            print(f"\n💾 Guardando batch {batch_number} en {output_file.name}...")
            
            with open(output_file, 'wb') as f:
                pickle.dump(current_batch, f)
            
            # Calcular tamaño del archivo
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"   ✅ Batch {batch_number} guardado: {len(current_batch)} traces, {file_size_mb:.2f} MB")
            
            # Resetear batch y liberar memoria
            current_batch = []
            batch_number += 1
            
            # Forzar garbage collection para liberar memoria
            import gc
            gc.collect()
    
    # Guardar el último batch si tiene datos
    if current_batch:
        output_file = output_dir / f"{model_name}_{dataset_name}_batch_{batch_number:04d}.pkl"
        print(f"\n💾 Guardando último batch {batch_number} en {output_file.name}...")
        
        with open(output_file, 'wb') as f:
            pickle.dump(current_batch, f)
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   ✅ Batch {batch_number} guardado: {len(current_batch)} traces, {file_size_mb:.2f} MB")
        batch_number += 1
    
    # Resumen final
    print("\n" + "="*80)
    print("✅ PROCESO COMPLETADO")
    print("="*80)
    print(f"Total de ejemplos procesados: {total_processed}")
    print(f"Total de errores: {total_errors}")
    print(f"Total de batches guardados: {batch_number}")
    print(f"Directorio de salida: {output_dir.absolute()}")
    
    # Listar archivos generados
    print(f"\n📁 Archivos generados:")
    batch_files = sorted(output_dir.glob(f"{model_name}_{dataset_name}_batch_*.pkl"))
    total_size = 0
    for batch_file in batch_files:
        size_mb = batch_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"   • {batch_file.name}: {size_mb:.2f} MB")
    
    print(f"\n💾 Tamaño total en disco: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    # Análisis básico del primer batch
    if batch_files:
        print(f"\n--- Análisis del primer batch ---")
        with open(batch_files[0], 'rb') as f:
            first_batch = pickle.load(f)
        
        if first_batch:
            sample_trace = first_batch[0]
            print(f"Estructura de cada trace:")
            print(f"  - Campos guardados: {list(sample_trace.keys())}")
            print(f"  - Número de capas: {len(sample_trace['hidden_states'])}")
            print(f"  - Total tokens en secuencia completa: {len(sample_trace['tokens'])}")
            
            if sample_trace['hidden_states']:
                hs_shape = sample_trace['hidden_states'][0].shape
                print(f"  - Shape de hidden state (capa 0): {hs_shape}")
                print(f"    → seq_len={hs_shape[0]} (prompt + respuesta), hidden_dim={hs_shape[1]}")
            
            if sample_trace['attentions']:
                attn_shape = sample_trace['attentions'][0].shape
                print(f"  - Shape de attention (capa 0): {attn_shape}")
                print(f"    → num_heads={attn_shape[0]}, seq_len={attn_shape[1]}x{attn_shape[2]}")
    
    return batch_number  # Retorna el número de batches creados


if __name__ == '__main__':
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(
        description="Extractor de trazas de activaciones y atenciones para modelos de lenguaje"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default="llama2_chat_7B",
        help='ID del modelo de HuggingFace (default: llama2_chat_7B)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['triviaqa', 'truthfulqa'],
        default='triviaqa',
        help='Dataset a utilizar: triviaqa o truthfulqa (default: triviaqa)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Número de muestras a procesar (default: None = todas)'
    )
    
    args = parser.parse_args()
    
    # Asignar model_id desde el argumento
    args.model_id = args.model
    
    seed_everything(SEED_VALUE)
    
    # --- Including the token to access to models if needed ---
    token_file = Path("llama_token.txt")
    if token_file.exists():
        with open(token_file, "r") as file:
            t = file.read().strip()
            login(token=t)
    
    traces = main(args)
