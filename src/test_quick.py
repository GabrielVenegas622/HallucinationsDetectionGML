"""
Script de prueba rápida para validar la extracción de trazas.
Procesa solo 3 ejemplos para verificar que todo funciona correctamente.
"""

import torch
import random
import os
import numpy as np
import pickle
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from datasets import load_dataset


def seed_everything(seed: int):
    """Fija todas las semillas para reproducibilidad."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def quick_test():
    """Prueba rápida con solo 3 ejemplos."""
    
    print("="*80)
    print("PRUEBA RÁPIDA DE EXTRACCIÓN DE TRAZAS")
    print("="*80)
    
    SEED_VALUE = 41
    seed_everything(SEED_VALUE)
    
    # Autenticación si es necesaria
    token_file = Path("llama_token.txt")
    if token_file.exists():
        with open(token_file, "r") as file:
            t = file.read().strip()
            login(token=t)
    
    # Modelo
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"\n1. Cargando modelo: {model_id}")
    print("   (Esto puede tomar unos minutos...)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    num_layers = len(model.model.layers)
    print(f"   ✅ Modelo cargado - {num_layers} capas")
    
    # Dataset
    print(f"\n2. Cargando TriviaQA...")
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    dataset = dataset.select(range(3))  # Solo 3 ejemplos
    print(f"   ✅ {len(dataset)} ejemplos seleccionados para prueba")
    
    # Procesar ejemplos
    print(f"\n3. Procesando ejemplos...")
    
    for idx, example in enumerate(dataset):
        question = example['question']
        
        print(f"\n   Ejemplo {idx + 1}/3:")
        print(f"   Q: {question[:70]}...")
        
        # Preparar prompt
        prompt_text = f"Answer the question concisely. Q: {question} A:"
        prompt = tokenizer(prompt_text, return_tensors='pt').to(model.device)
        
        # Generar
        with torch.no_grad():
            generation_output = model.generate(
                **prompt,
                num_beams=5,
                num_return_sequences=1,
                do_sample=False,
                max_new_tokens=32,  # Reducido para prueba
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_attentions=True,
                output_hidden_states=True
            )
        
        # Decodificar
        prompt_length = prompt["input_ids"].shape[-1]
        generated_answer = tokenizer.decode(
            generation_output.sequences[0, prompt_length:],
            skip_special_tokens=True
        )
        
        print(f"   A: {generated_answer[:70]}...")
        
        # Verificar dimensiones
        num_generated_tokens = len(generation_output.hidden_states)
        num_layers_check = len(generation_output.hidden_states[0]) - 1
        
        print(f"   ✅ Capturados: {num_generated_tokens} tokens, {num_layers_check} capas")
        
        # Verificar shapes
        if generation_output.hidden_states:
            first_hidden = generation_output.hidden_states[0][1]  # Primera capa (skip embeddings)
            print(f"   ✅ Hidden state shape: {first_hidden.shape}")
        
        if generation_output.attentions:
            first_attn = generation_output.attentions[0][0]  # Primera capa
            print(f"   ✅ Attention shape: {first_attn.shape}")
    
    print("\n" + "="*80)
    print("✅ PRUEBA COMPLETADA EXITOSAMENTE")
    print("="*80)
    print("\nPróximos pasos:")
    print("  1. Ejecutar: python src/trace_extractor.py")
    print("  2. Inspeccionar: python src/inspect_traces.py")
    print("")


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()
