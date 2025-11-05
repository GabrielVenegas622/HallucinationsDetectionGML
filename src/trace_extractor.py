import torch
import random
import os
import numpy as np
import pickle
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


def find_answer_cutoff_point(text, tokenizer, token_ids, prompt_length):
    """
    Encuentra el punto de corte óptimo para la respuesta.
    Intenta múltiples estrategias para detectar dónde termina la respuesta real.
    
    Args:
        text: Texto generado completo
        tokenizer: Tokenizer del modelo
        token_ids: IDs de tokens generados (numpy array)
        prompt_length: Longitud del prompt en tokens
        
    Returns:
        tuple: (cutoff_token_index, cutoff_method)
            cutoff_token_index: índice del token donde cortar (relativo al inicio de la generación)
            cutoff_method: string indicando qué método se usó
    """
    # Estrategia 1: Buscar el primer punto seguido de espacio o final
    first_period_idx = text.find('.')
    if first_period_idx != -1:
        # Encontrar en qué token está el punto
        text_before_period = text[:first_period_idx + 1]
        tokens_before = tokenizer.encode(text_before_period, add_special_tokens=False)
        return len(tokens_before), "first_period"
    
    # Estrategia 2: Buscar salto de línea (común cuando el modelo empieza a divagar)
    first_newline_idx = text.find('\n')
    if first_newline_idx != -1:
        text_before_newline = text[:first_newline_idx]
        tokens_before = tokenizer.encode(text_before_newline, add_special_tokens=False)
        return len(tokens_before), "first_newline"

    # Estrategia 2.1: Buscar un punto + salto de línea!. (Común en Llama).
    first_periodnew_idx=text.find('.\n')
    if first_periodnew_idx != -1:
        text_before_periodnew = text[:first_periodnew_idx]
        tokens_before = tokenizer.encode(text_before_periodnew, add_special_tokens=False)
        return len(tokens_before), "first_periodnewline"

    
    # Estrategia 3: Buscar otros signos de puntuación finales (?, !)
    for punct, method in [('?', 'question_mark'), ('!', 'exclamation')]:
        idx = text.find(punct)
        if idx != -1:
            text_before = text[:idx + 1]
            tokens_before = tokenizer.encode(text_before, add_special_tokens=False)
            return len(tokens_before), method
    
    # Estrategia 4: Detectar repetición (común en generaciones redundantes)
    words = text.split()
    if len(words) > 10:
        # Buscar secuencias repetidas
        for i in range(len(words) - 3):
            window = ' '.join(words[i:i+3])
            rest = ' '.join(words[i+3:])
            if window in rest:
                # Hay repetición, cortar antes
                text_before_repeat = ' '.join(words[:i])
                tokens_before = tokenizer.encode(text_before_repeat, add_special_tokens=False)
                return max(1, len(tokens_before)), "repetition_detected"
    
    # Estrategia 5: Si todo falla, usar toda la generación
    return len(token_ids) - prompt_length, "full_generation"


def extract_activations_and_attentions(model, tokenizer, question, answer=None, 
                                       max_new_tokens=64, cut_at_period=True):
    """
    Extrae las activaciones (hidden states) y atenciones del estado final
    de la generación (después de generar todos los tokens).
    
    Args:
        model: Modelo de lenguaje cargado
        tokenizer: Tokenizer correspondiente al modelo
        question: Pregunta a responder
        answer: Respuesta de referencia (opcional, para evaluación)
        max_new_tokens: Número máximo de tokens a generar
        cut_at_period: Si True, corta las trazas en el primer punto de la respuesta
        
    Returns:
        dict con:
            - 'generated_answer_clean': respuesta cortada en punto/señal de fin
            - 'hidden_states': lista de arrays por capa [seq_len_total, hidden_dim]
            - 'attentions': lista de arrays por capa [num_heads, seq_len_total, seq_len_total]
            - 'tokens': tokens generados (IDs)
            - 'tokens_decoded': tokens como strings
            
        donde seq_len_total = len(prompt) + len(respuesta_generada_limpia)
    """
    # Preparar el prompt (compatible con Qwen y Llama)
    # Para Llama, usar formato de chat si el tokenizer lo soporta
    if hasattr(tokenizer, 'apply_chat_template') and 'llama' in tokenizer.name_or_path.lower():
        messages = [
            {"role": "user", "content": f"Answer the following question concisely in one sentence:\n\n{question}"}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Formato genérico para Qwen y otros
        prompt_text = f"Answer the question concisely in one sentence.\n\nQuestion: {question}\nAnswer:"
    
    prompt = tokenizer(prompt_text, return_tensors='pt').to(model.device)
    
    # Asegurar que el tokenizer tiene tokens especiales configurados
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    
    # Generar con activaciones y atenciones
    # Usar parámetros más agresivos para forzar respuestas cortas
    with torch.no_grad():
        generation_output = model.generate(
            **prompt,
            num_beams=5,
            num_return_sequences=1,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.5,  # Aumentado para evitar repetición
            length_penalty=0.8,      # Penalizar respuestas largas
            no_repeat_ngram_size=3,  # Evitar repetición de 3-gramas
            early_stopping=True,     # Detener en EOS
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_attentions=True,
            output_hidden_states=True
        )
    
    # Decodificar el texto generado
    generated_ids = generation_output.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    prompt_length = prompt["input_ids"].shape[-1]
    generated_answer = tokenizer.decode(
        generation_output.sequences[0, prompt_length:],
        skip_special_tokens=True
    )
    
    # Determinar punto de corte
    cutoff_token_count, cutoff_method = find_answer_cutoff_point(
        generated_answer, tokenizer, generated_ids, prompt_length
    )
    
    # Aplicar corte si está habilitado
    if cut_at_period and cutoff_token_count < len(generated_ids) - prompt_length:
        # Cortar en el punto detectado
        actual_tokens_to_use = cutoff_token_count
        generated_answer_clean = tokenizer.decode(
            generation_output.sequences[0, prompt_length:prompt_length + actual_tokens_to_use],
            skip_special_tokens=True
        )
    else:
        actual_tokens_to_use = len(generated_ids) - prompt_length
        generated_answer_clean = generated_answer
        cutoff_method = "no_cutoff_applied"
    
    # Extraer hidden states y attentions del ESTADO FINAL (último paso de generación)
    num_generated_tokens = len(generation_output.hidden_states)
    num_layers = len(generation_output.hidden_states[0]) - 1  # -1 porque incluye embeddings
    
    # Limitar al número de tokens que realmente queremos
    tokens_to_extract = min(actual_tokens_to_use, num_generated_tokens)
    
    # Índice del último paso de generación
    final_step_idx = tokens_to_extract - 1
    
    # Longitud total de la secuencia (prompt + tokens generados)
    seq_len_total = prompt_length + tokens_to_extract
    
    # Organizar hidden states: [capa] -> array[seq_len_total, hidden_dim]
    # Extraemos SOLO el último paso que contiene el estado completo
    hidden_states_by_layer = []
    for layer_idx in range(1, num_layers + 1):  # Empezar desde 1 para saltar embeddings
        # Estado final de esta capa (incluye prompt + todos los tokens generados)
        final_state = generation_output.hidden_states[final_step_idx][layer_idx]
        # Cortar para incluir solo: prompt + tokens generados hasta el corte
        # Shape: [batch=1, seq_len, hidden_dim] -> [seq_len_total, hidden_dim]
        final_state_cut = final_state[0, :seq_len_total, :].cpu().numpy()
        hidden_states_by_layer.append(final_state_cut)
    
    # Organizar attentions: [capa] -> array[num_heads, seq_len_total, seq_len_total]
    # Extraemos SOLO el último paso que contiene la matriz de atención completa
    attentions_by_layer = []
    for layer_idx in range(num_layers):
        # Atenciones finales de esta capa
        final_attn = generation_output.attentions[final_step_idx][layer_idx]
        # Cortar matriz de atención a tamaño correcto
        # Shape: [batch=1, num_heads, seq_len, seq_len] -> [num_heads, seq_len_total, seq_len_total]
        final_attn_cut = final_attn[0, :, :seq_len_total, :seq_len_total].cpu().numpy()
        attentions_by_layer.append(final_attn_cut)
    
    # Tokens cortados (solo la parte generada, sin prompt)
    generated_tokens_clean = generation_output.sequences[0, prompt_length:prompt_length + actual_tokens_to_use].cpu().numpy()
    
    # Decodificar cada token individualmente para visualización
    tokens_decoded = []
    for token_id in generated_tokens_clean:
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        tokens_decoded.append(token_text)
    
    return {
        'hidden_states': hidden_states_by_layer,  # [num_layers] cada uno: [seq_len_total, hidden_dim]
        'attentions': attentions_by_layer,  # [num_layers] cada uno: [num_heads, seq_len_total, seq_len_total]
        'tokens': generated_tokens_clean,  # IDs de tokens (solo respuesta, sin prompt)
        'tokens_decoded': tokens_decoded,  # Textos de tokens decodificados
        'generated_answer_clean': generated_answer_clean  # Respuesta limpia
    }


def main():
    # Configuración del modelo
    # Opciones:
    # - "meta-llama/Llama-2-7b-chat-hf" (recomendado para alucinaciones)
    # - "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" (pre-cuantizado, pero rechaza muchas preguntas)
    # agregando mini commit
    model_id = "meta-llama/Llama-2-7b-chat-hf"

    # Configuración de batches para gestión de memoria
    BATCH_SIZE = 500
    
    print(f"Cargando modelo: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Detectar si el modelo ya está cuantizado (bnb-4bit o bnb-8bit en el nombre)
    is_prequantized = "bnb-4bit" in model_id.lower() or "bnb-8bit" in model_id.lower()
    
    if is_prequantized:
        print("✅ Modelo pre-cuantizado detectado, cargando directamente...")
        # Para modelos pre-cuantizados de Unsloth, NO usar quantization_config
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
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
            model_id,
            attn_implementation="eager",  # Necesario para extraer atenciones
            quantization_config=bnb_config,
            device_map="auto",            # Distribución automática en GPU
            torch_dtype=torch.float16     # Tipo de dato base
        )
    
    num_layers = len(model.model.layers)
    print(f"Número de capas del modelo: {num_layers}")
    
    # Cargar dataset TriviaQA
    print("\nCargando dataset TriviaQA...")
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    
    # Limitar el número de ejemplos si se desea (None = procesar todo el dataset)
    num_samples = None  # Cambiar a un número específico para limitar, ej: 1000
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
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
    
    # Estadísticas de corte
    cutoff_stats = {}
    
    # Procesar cada ejemplo del dataset
    for idx, example in enumerate(tqdm(dataset, desc="Extrayendo trazas")):
        question = example['question']
        question_id = example['question_id']  # ID único de TriviaQA
        
        try:
            # Extraer activaciones y atenciones con corte automático
            traces = extract_activations_and_attentions(
                model=model,
                tokenizer=tokenizer,
                question=question,
                answer=None,
                max_new_tokens=64,
                cut_at_period=True  # Activar corte inteligente
            )
            
            # Añadir solo el question_id como metadata
            traces['question_id'] = question_id
            
            current_batch.append(traces)
            total_processed += 1
            
            # Actualizar estadísticas de corte (para debugging, no se guarda)
            num_tokens = len(traces['tokens'])
            cutoff_stats['processed'] = cutoff_stats.get('processed', 0) + 1
            
            # Mostrar ejemplo cada 10 muestras
            if idx % 10 == 0:
                print(f"\n--- Ejemplo {idx} (Batch actual: {len(current_batch)}/{BATCH_SIZE}) ---")
                print(f"Question ID: {question_id}")
                print(f"Pregunta: {question}...")
                print(f"Respuesta limpia: {traces['generated_answer_clean']}...")
                print(f"Tokens generados: {num_tokens}")
                print(f"Tokens decodificados: {traces['tokens_decoded'][:5]}...")  # Primeros 5 tokens
            
        except Exception as e:
            print(f"\n⚠️  Error procesando ejemplo {idx}: {e}")
            total_errors += 1
            continue
        
        # Guardar batch cuando alcance el tamaño especificado
        if len(current_batch) >= BATCH_SIZE:
            output_file = output_dir / f"trivia_qa_traces_batch_{batch_number:04d}.pkl"
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
        output_file = output_dir / f"trivia_qa_traces_batch_{batch_number:04d}.pkl"
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
    batch_files = sorted(output_dir.glob("trivia_qa_traces_batch_*.pkl"))
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
            print(f"  - Tokens en respuesta: {len(sample_trace['tokens'])}")
            
            if sample_trace['hidden_states']:
                first_layer_first_token = sample_trace['hidden_states'][0][0]
                print(f"  - Dimensión de hidden state (primera capa, primer token): {first_layer_first_token.shape}")
            
            if sample_trace['attentions']:
                first_layer_first_token_attn = sample_trace['attentions'][0][0]
                print(f"  - Dimensión de attention (primera capa, primer token): {first_layer_first_token_attn.shape}")
    
    return batch_number  # Retorna el número de batches creados


if __name__ == '__main__':
    seed_everything(SEED_VALUE)
    # --- Including the token to access to models if needed ---
    token_file = Path("llama_token.txt")
    if token_file.exists():
        with open(token_file, "r") as file:
            t = file.read().strip()
            login(token=t)
    
    traces = main()
