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


def extract_activations_and_attentions(model, tokenizer, question, answer=None, max_new_tokens=64):
    """
    Extrae las activaciones (hidden states) y atenciones de todas las capas
    durante la generación de una respuesta.
    
    Args:
        model: Modelo de lenguaje cargado
        tokenizer: Tokenizer correspondiente al modelo
        question: Pregunta a responder
        answer: Respuesta de referencia (opcional, para evaluación)
        max_new_tokens: Número máximo de tokens a generar
        
    Returns:
        dict con:
            - 'question': pregunta original
            - 'generated_text': texto generado completo
            - 'generated_answer': respuesta sin el prompt
            - 'hidden_states': lista de hidden states por capa y token
            - 'attentions': lista de matrices de atención por capa y token
            - 'tokens': tokens generados (IDs)
    """
    # Preparar el prompt
    prompt_text = f"Answer the question concisely. Q: {question} A:"
    prompt = tokenizer(prompt_text, return_tensors='pt').to(model.device)
    
    # Generar con activaciones y atenciones
    with torch.no_grad():
        generation_output = model.generate(
            **prompt,
            num_beams=5,
            num_return_sequences=1,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
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
    
    # Extraer hidden states y attentions
    # generation_output.hidden_states es una tupla de longitud igual al número de tokens generados
    # Cada elemento es una tupla de tensores (uno por capa + embedding)
    # generation_output.attentions tiene la misma estructura pero para atenciones
    
    num_generated_tokens = len(generation_output.hidden_states)
    num_layers = len(generation_output.hidden_states[0]) - 1  # -1 porque incluye embeddings
    
    # Organizar hidden states: [capa][paso_generación][batch, seq_len, hidden_dim]
    hidden_states_by_layer = []
    for layer_idx in range(1, num_layers + 1):  # Empezar desde 1 para saltar embeddings
        layer_states = []
        for token_step in range(num_generated_tokens):
            # Extraer el hidden state de esta capa en este paso
            state = generation_output.hidden_states[token_step][layer_idx]
            layer_states.append(state.cpu().numpy())
        hidden_states_by_layer.append(layer_states)
    
    # Organizar attentions: [capa][paso_generación][batch, num_heads, seq_len, seq_len]
    attentions_by_layer = []
    for layer_idx in range(num_layers):
        layer_attns = []
        for token_step in range(num_generated_tokens):
            # Extraer la matriz de atención de esta capa en este paso
            attn = generation_output.attentions[token_step][layer_idx]
            layer_attns.append(attn.cpu().numpy())
        attentions_by_layer.append(layer_attns)
    
    return {
        'question': question,
        'generated_text': generated_text,
        'generated_answer': generated_answer,
        'hidden_states': hidden_states_by_layer,  # [num_layers][num_tokens_generated]
        'attentions': attentions_by_layer,  # [num_layers][num_tokens_generated]
        'tokens': generated_ids.cpu().numpy(),
        'prompt_length': prompt_length,
        'num_layers': num_layers
    }


def main():
    # Configuración del modelo Qwen3-4B-Instruct
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    
    print(f"Cargando modelo: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",  # Necesario para extraer atenciones
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    num_layers = len(model.model.layers)
    print(f"Número de capas del modelo: {num_layers}")
    
    # Cargar dataset TriviaQA
    print("\nCargando dataset TriviaQA...")
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    
    # Limitar el número de ejemplos para prueba (puedes ajustar según necesites)
    num_samples = 1  # Cambiar según necesidad
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"Número de muestras a procesar: {len(dataset)}")
    
    # Crear directorio para guardar los traces
    output_dir = Path("./traces_data")
    output_dir.mkdir(exist_ok=True)
    
    # Procesar cada ejemplo del dataset
    all_traces = []
    
    for idx, example in enumerate(tqdm(dataset, desc="Extrayendo trazas")):
        question = example['question']
        # TriviaQA tiene múltiples respuestas posibles
        answer_aliases = example['answer']['aliases'] if 'answer' in example else None
        
        try:
            # Extraer activaciones y atenciones
            traces = extract_activations_and_attentions(
                model=model,
                tokenizer=tokenizer,
                question=question,
                answer=answer_aliases,
                max_new_tokens=64
            )
            
            # Añadir metadata del ejemplo
            traces['example_id'] = idx
            traces['ground_truth_answers'] = answer_aliases
            
            all_traces.append(traces)
            
            # Mostrar ejemplo cada 10 muestras
            if idx % 10 == 0:
                print(f"\n--- Ejemplo {idx} ---")
                print(f"Pregunta: {question}")
                print(f"Respuesta generada: {traces['generated_answer'][:100]}...")
                print(f"Número de capas: {traces['num_layers']}")
                print(f"Tokens generados: {len(traces['tokens']) - traces['prompt_length']}")
            
        except Exception as e:
            print(f"\nError procesando ejemplo {idx}: {e}")
            continue
    
    # Guardar todos los traces
    output_file = output_dir / f"trivia_qa_traces_{model_id.split('/')[-1]}.pkl"
    print(f"\nGuardando traces en {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(all_traces, f)
    
    print(f"✅ Proceso completado. {len(all_traces)} ejemplos procesados y guardados.")
    
    # Análisis básico de los datos extraídos
    print("\n--- Resumen de datos extraídos ---")
    if all_traces:
        sample_trace = all_traces[0]
        print(f"Estructura de cada trace:")
        print(f"  - Número de capas: {sample_trace['num_layers']}")
        print(f"  - Hidden states shape por capa: {len(sample_trace['hidden_states'])} capas")
        print(f"  - Attentions shape por capa: {len(sample_trace['attentions'])} capas")
        
        # Mostrar dimensiones de un ejemplo
        if sample_trace['hidden_states']:
            first_layer_first_token = sample_trace['hidden_states'][0][0]
            print(f"  - Dimensión de hidden state (primera capa, primer token): {first_layer_first_token.shape}")
        
        if sample_trace['attentions']:
            first_layer_first_token_attn = sample_trace['attentions'][0][0]
            print(f"  - Dimensión de attention (primera capa, primer token): {first_layer_first_token_attn.shape}")
    
    return all_traces


if __name__ == '__main__':
    seed_everything(SEED_VALUE)
    # --- Including the token to access to models if needed ---
    token_file = Path("llama_token.txt")
    if token_file.exists():
        with open(token_file, "r") as file:
            t = file.read().strip()
            login(token=t)
    
    traces = main()
