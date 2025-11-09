"""
Script para generar ground truth con scores BLEURT para traces.

Este script:
1. Carga los traces generados por trace_extractor.py desde archivos .pkl
2. Carga el dataset original (TriviaQA o TruthfulQA) para obtener respuestas correctas
3. Calcula el score BLEURT entre la respuesta generada y la respuesta correcta
4. Genera un archivo CSV con pares (question_id, bleurt_score)

Uso:
    python trace_to_gt.py --dataset triviaqa --traces-dir ./traces_data --output ground_truth_scores.csv
    python trace_to_gt.py --dataset truthfulqa --traces-dir ./traces_data --output ground_truth_scores.csv
"""

import pickle
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import torch


def load_bleurt_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Carga el modelo BLEURT desde HuggingFace.
    
    Usamos la implementación oficial de BLEURT adaptada para HuggingFace.
    Modelo: lucadiliello/BLEURT-20
    
    Args:
        device: 'cuda' o 'cpu'
        
    Returns:
        tuple: (tokenizer, model)
    """
    print(f"Cargando modelo BLEURT en {device}...")
    
    # Usar la implementación de BLEURT disponible en HuggingFace
    model_name = 'lucadiliello/BLEURT-20'
    

    config = BleurtConfig.from_pretrained(model_name)
    tokenizer = BleurtTokenizer.from_pretrained(model_name)
    model = BleurtForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    print("✅ Modelo BLEURT cargado correctamente")
    return tokenizer, model

def compute_max_bleurt_score(reference_answers, generated_answer, tokenizer, model, device='cuda'):
    """
    Calcula el score BLEURT máximo comparando la respuesta generada con TODAS
    las respuestas de referencia disponibles.
    
    Args:
        reference_answers: Lista de respuestas correctas (ground truth)
        generated_answer: Respuesta generada por el modelo
        tokenizer: Tokenizer de BLEURT
        model: Modelo BLEURT
        device: Dispositivo para cómputo
        
    Returns:
        tuple: (max_score, best_reference, all_scores)
            - max_score: El score BLEURT máximo obtenido
            - best_reference: La respuesta de referencia que dio el mejor score
            - all_scores: Lista con todos los scores calculados
    """
    if not reference_answers:
        return 0.0, '', []
    
    all_scores = []
    generated_answers = [generated_answer]*len(reference_answers)
    with torch.no_grad():
        # Tokenizar el par referencia-candidato
        inputs = tokenizer(
            reference_answers, 
            generated_answers, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=128
        ).to(device)
        
        # Obtener el score
        outputs = model(**inputs)
        all_scores = outputs.logits.flatten().tolist()

    
    # Encontrar el score máximo y su referencia correspondiente
    max_score = max(all_scores)
    max_idx = all_scores.index(max_score)
    best_reference = reference_answers[max_idx]
    print(max_score, best_reference, all_scores)
    return max_score, best_reference, all_scores


def load_ground_truth_dataset(dataset_name):
    """
    Carga el dataset original para obtener TODAS las respuestas correctas.
    
    Args:
        dataset_name: 'triviaqa' o 'truthfulqa'
        
    Returns:
        dict: Mapeo de question_id -> lista de ground_truth_answers
    """
    print(f"\nCargando dataset {dataset_name} para ground truth...")
    
    ground_truth = {}
    
    if dataset_name.lower() == 'triviaqa':
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        
        for idx, example in enumerate(tqdm(dataset, desc="Cargando TriviaQA")):
            question_id = example['question_id']
            # TriviaQA tiene múltiples respuestas correctas en 'answer'
            # Guardamos TODAS las respuestas normalizadas
            answers_list = []
            
            if 'answer' in example:
                # Intentar obtener normalized_aliases (lista de variantes aceptadas)
                if 'normalized_aliases' in example['answer']:
                    normalized = example['answer']['normalized_aliases']
                    if normalized:
                        answers_list.extend(normalized)
                
                # También incluir el valor principal si existe
                if 'value' in example['answer'] and example['answer']['value']:
                    if example['answer']['value'] not in answers_list:
                        answers_list.append(example['answer']['value'])
                
                # Incluir aliases si existen
                if 'aliases' in example['answer']:
                    aliases = example['answer']['aliases']
                    if aliases:
                        for alias in aliases:
                            if alias not in answers_list:
                                answers_list.append(alias)
            
            # Si no encontramos ninguna respuesta, usar lista vacía
            ground_truth[question_id] = answers_list if answers_list else ['']
    
    elif dataset_name.lower() == 'truthfulqa':
        dataset = load_dataset("truthful_qa", "generation", split="validation")
        
        for idx, example in enumerate(tqdm(dataset, desc="Cargando TruthfulQA")):
            # TruthfulQA no tiene question_id, usar índice
            question_id = f"truthfulqa_{idx}"
            
            answers_list = []
            
            # TruthfulQA tiene 'correct_answers' que es una lista de respuestas correctas
            if 'correct_answers' in example and example['correct_answers']:
                answers_list.extend(example['correct_answers'])
            
            # También incluir 'best_answer' si existe y no está en la lista
            if 'best_answer' in example and example['best_answer']:
                if example['best_answer'] not in answers_list:
                    answers_list.append(example['best_answer'])
            
            # Si no encontramos ninguna respuesta, usar lista vacía
            ground_truth[question_id] = answers_list if answers_list else ['']
    
    else:
        raise ValueError(f"Dataset no soportado: {dataset_name}")
    
    # Calcular estadísticas de respuestas
    total_answers = sum(len(answers) for answers in ground_truth.values())
    avg_answers = total_answers / len(ground_truth) if ground_truth else 0
    
    print(f"✅ Cargadas {len(ground_truth)} preguntas con ground truth")
    print(f"   Total de respuestas: {total_answers}")
    print(f"   Promedio de respuestas por pregunta: {avg_answers:.2f}")
    
    return ground_truth


def load_traces_from_pkl(traces_dir, dataset_name):
    """
    Carga todos los traces desde archivos .pkl en el directorio.
    
    Args:
        traces_dir: Path al directorio con archivos .pkl
        dataset_name: Nombre del dataset para filtrar archivos
        
    Returns:
        dict: Mapeo de question_id -> trace_data
    """
    traces_dir = Path(traces_dir)
    
    # Buscar archivos .pkl que correspondan al dataset
    pkl_pattern = f"*{dataset_name}*.pkl"
    pkl_files = sorted(traces_dir.glob(pkl_pattern))
    
    if not pkl_files:
        print(f"⚠️  No se encontraron archivos .pkl para {dataset_name} en {traces_dir}")
        print(f"   Patrón de búsqueda: {pkl_pattern}")
        return {}
    
    print(f"\nCargando traces desde {len(pkl_files)} archivos .pkl...")
    
    traces_dict = {}
    
    for pkl_file in tqdm(pkl_files, desc="Cargando archivos"):
        with open(pkl_file, 'rb') as f:
            batch_traces = pickle.load(f)
        
        for trace in batch_traces:
            question_id = trace['question_id']
            traces_dict[question_id] = trace
    
    print(f"✅ Cargados {len(traces_dict)} traces únicos")
    return traces_dict


def generate_ground_truth_scores(args):
    """
    Función principal que genera el archivo con scores BLEURT.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("="*80)
    print("GENERACIÓN DE GROUND TRUTH CON SCORES BLEURT")
    print("="*80)
    
    # 1. Cargar modelo BLEURT
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer, model = load_bleurt_model(device)
    
    # 2. Cargar ground truth del dataset original
    ground_truth = load_ground_truth_dataset(args.dataset)
    
    # 3. Cargar traces generados
    traces = load_traces_from_pkl(args.traces_dir, args.dataset)
    
    if not traces:
        print("❌ No se encontraron traces para procesar")
        return
    
    # 4. Calcular scores BLEURT
    print("\n" + "="*80)
    print("CALCULANDO SCORES BLEURT (MAX sobre todas las referencias)")
    print("="*80)
    
    results = []
    missing_ground_truth = 0
    total_comparisons = 0
    
    for question_id, trace in tqdm(traces.items(), desc="Procesando traces"):
        # Obtener respuesta generada
        generated_answer = trace.get('generated_answer_clean', '')
        
        # Obtener respuestas de ground truth (lista)
        if question_id not in ground_truth:
            print(f"⚠️  No se encontró ground truth para question_id: {question_id}")
            missing_ground_truth += 1
            continue
        
        reference_answers = ground_truth[question_id]
        
        # Calcular score BLEURT máximo comparando con TODAS las referencias
        print(f"Respuesta: {generated_answer}, GT: {reference_answers}")
        max_score, best_reference, all_scores = compute_max_bleurt_score(
            reference_answers, 
            generated_answer, 
            tokenizer, 
            model, 
            device
        )
        
        total_comparisons += len(all_scores)
        
        results.append({
            'question_id': question_id,
            'bleurt_score': max_score,
            'generated_answer': generated_answer,
            'best_reference': best_reference,
            'num_references': len(reference_answers),
            'all_scores': all_scores,
            'min_score': min(all_scores) if all_scores else 0.0,
            'avg_score': sum(all_scores) / len(all_scores) if all_scores else 0.0
        })
    
    # 5. Crear DataFrame y guardar resultados
    df = pd.DataFrame(results)
    
    # Ordenar por question_id
    df = df.sort_values('question_id')
    
    # Guardar archivo completo con todas las columnas
    full = args.output.split('.')[0] + '_full'
    output_path_full = Path(full).with_suffix('.csv')
    df_full = df.copy()
    # Convertir la lista de scores a string para guardarla en CSV
    df_full['all_scores'] = df_full['all_scores'].apply(lambda x: ';'.join([f'{s:.4f}' for s in x]))
    df_full.to_csv(output_path_full, index=False)
    print(f"\n✅ Archivo completo guardado: {output_path_full}")
    
    # Guardar archivo simplificado solo con id y score
    df_simple = df[['question_id', 'bleurt_score']]
    df_simple.to_csv(args.output, index=False)
    print(f"✅ Archivo de scores guardado: {args.output}")
    
    # 6. Estadísticas
    print("\n" + "="*80)
    print("ESTADÍSTICAS")
    print("="*80)
    print(f"Total de traces procesados: {len(results)}")
    print(f"Traces sin ground truth: {missing_ground_truth}")
    print(f"Total de comparaciones realizadas: {total_comparisons}")
    print(f"Promedio de referencias por pregunta: {df['num_references'].mean():.2f}")
    print(f"\nEstadísticas de BLEURT scores (MAX):")
    print(f"  Media: {df['bleurt_score'].mean():.4f}")
    print(f"  Mediana: {df['bleurt_score'].median():.4f}")
    print(f"  Desviación estándar: {df['bleurt_score'].std():.4f}")
    print(f"  Mínimo: {df['bleurt_score'].min():.4f}")
    print(f"  Máximo: {df['bleurt_score'].max():.4f}")
    
    # Estadísticas de diferencia entre max y avg
    df['score_range'] = df['bleurt_score'] - df['min_score']
    print(f"\nDiferencia entre MAX y MIN score por pregunta:")
    print(f"  Promedio de rango: {df['score_range'].mean():.4f}")
    print(f"  Máximo rango: {df['score_range'].max():.4f}")
    
    # Mostrar algunos ejemplos
    print("\n" + "="*80)
    print("EJEMPLOS DE SCORES")
    print("="*80)
    
    # Mejor score
    best_idx = df['bleurt_score'].idxmax()
    print(f"\n🏆 MEJOR SCORE ({df.loc[best_idx, 'bleurt_score']:.4f}):")
    print(f"   Question ID: {df.loc[best_idx, 'question_id']}")
    print(f"   Generado: {df.loc[best_idx, 'generated_answer'][:100]}...")
    print(f"   Mejor Referencia: {df.loc[best_idx, 'best_reference'][:100]}...")
    print(f"   Referencias evaluadas: {df.loc[best_idx, 'num_references']}")
    print(f"   Rango de scores: [{df.loc[best_idx, 'min_score']:.4f}, {df.loc[best_idx, 'bleurt_score']:.4f}]")
    
    # Peor score
    worst_idx = df['bleurt_score'].idxmin()
    print(f"\n⚠️  PEOR SCORE ({df.loc[worst_idx, 'bleurt_score']:.4f}):")
    print(f"   Question ID: {df.loc[worst_idx, 'question_id']}")
    print(f"   Generado: {df.loc[worst_idx, 'generated_answer'][:100]}...")
    print(f"   Mejor Referencia: {df.loc[worst_idx, 'best_reference'][:100]}...")
    print(f"   Referencias evaluadas: {df.loc[worst_idx, 'num_references']}")
    print(f"   Rango de scores: [{df.loc[worst_idx, 'min_score']:.4f}, {df.loc[worst_idx, 'bleurt_score']:.4f}]")
    
    # Mayor beneficio de múltiples referencias
    max_range_idx = df['score_range'].idxmax()
    print(f"\n📊 MAYOR BENEFICIO DE MÚLTIPLES REFERENCIAS ({df.loc[max_range_idx, 'score_range']:.4f}):")
    print(f"   Question ID: {df.loc[max_range_idx, 'question_id']}")
    print(f"   Score MAX: {df.loc[max_range_idx, 'bleurt_score']:.4f}")
    print(f"   Score MIN: {df.loc[max_range_idx, 'min_score']:.4f}")
    print(f"   Score AVG: {df.loc[max_range_idx, 'avg_score']:.4f}")
    print(f"   Referencias evaluadas: {df.loc[max_range_idx, 'num_references']}")
    
    print("\n" + "="*80)
    print("✅ PROCESO COMPLETADO")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Genera ground truth con scores BLEURT para traces"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['triviaqa', 'truthfulqa'],
        required=True,
        help='Dataset utilizado: triviaqa o truthfulqa'
    )
    
    parser.add_argument(
        '--traces-dir',
        type=str,
        default='./traces_data',
        help='Directorio con archivos .pkl de traces (default: ./traces_data)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='ground_truth_scores.csv',
        help='Archivo de salida CSV (default: ground_truth_scores.csv)'
    )
    
    args = parser.parse_args()
    
    generate_ground_truth_scores(args)
