#!/usr/bin/env python3
"""
Inspección Detallada de Traces

Analiza la estructura de los traces para entender el mismatch entre
hidden_states y attentions.

Uso:
    python inspect_trace_structure.py --data-pattern "traces_data/*.pkl" --num-samples 5
"""

import pickle
import glob
import numpy as np
import argparse


def inspect_trace_structure(file_pattern, num_samples=5):
    """Inspecciona la estructura detallada de los traces"""
    
    print("="*80)
    print("INSPECCIÓN DETALLADA DE TRACES")
    print("="*80)
    
    files = glob.glob(file_pattern)
    if not files:
        print(f"❌ No se encontraron archivos con patrón: {file_pattern}")
        return
    
    print(f"\nEncontrados {len(files)} archivos .pkl")
    print(f"Inspeccionando {num_samples} traces de ejemplo...\n")
    
    all_samples = []
    
    # Recolectar muestras
    for file_path in files[:min(3, len(files))]:
        try:
            with open(file_path, 'rb') as f:
                traces = pickle.load(f)
            
            for i, trace in enumerate(traces[:min(2, len(traces))]):
                all_samples.append((file_path, i, trace))
                if len(all_samples) >= num_samples:
                    break
        except Exception as e:
            print(f"❌ Error al cargar {file_path}: {e}")
        
        if len(all_samples) >= num_samples:
            break
    
    # Analizar muestras
    for sample_idx, (file_path, trace_idx, trace) in enumerate(all_samples):
        print("="*80)
        print(f"MUESTRA {sample_idx + 1}")
        print("="*80)
        print(f"Archivo: {file_path}")
        print(f"Trace índice: {trace_idx}")
        print(f"Question ID: {trace.get('question_id', 'N/A')}")
        print(f"Generated answer: {trace.get('generated_answer_clean', 'N/A')[:100]}...")
        
        # Información de tokens
        if 'tokens' in trace:
            tokens = trace['tokens']
            print(f"\n📊 TOKENS COMPLETOS (prompt + respuesta):")
            print(f"   Total tokens: {len(tokens)}")
            print(f"   Tipo: {type(tokens)}, dtype: {tokens.dtype if hasattr(tokens, 'dtype') else 'N/A'}")
        
        if 'tokens_decoded' in trace:
            tokens_decoded = trace['tokens_decoded']
            print(f"   Tokens decoded disponibles: {len(tokens_decoded)}")
            print(f"   Primeros 5 tokens: {tokens_decoded[:5]}")
            print(f"   Últimos 5 tokens: {tokens_decoded[-5:]}")
        
        # Información de hidden states
        num_layers = len(trace['hidden_states'])
        print(f"\n🧠 HIDDEN STATES:")
        print(f"   Número de capas: {num_layers}")
        
        for layer_idx in [0, num_layers//2, num_layers-1]:
            hs = trace['hidden_states'][layer_idx]
            print(f"   Capa {layer_idx:2d}: shape={hs.shape}, "
                  f"dtype={hs.dtype}, "
                  f"range=[{hs.min():.4f}, {hs.max():.4f}]")
        
        # Información de attentions
        print(f"\n👁️  ATTENTIONS:")
        for layer_idx in [0, num_layers//2, num_layers-1]:
            attn = trace['attentions'][layer_idx]
            print(f"   Capa {layer_idx:2d}: shape={attn.shape}, "
                  f"dtype={attn.dtype}, "
                  f"range=[{attn.min():.4f}, {attn.max():.4f}]")
        
        # Análisis crítico: ¿coinciden las dimensiones?
        print(f"\n🔍 ANÁLISIS DE COINCIDENCIA:")
        
        critical_issues = []
        warnings = []
        
        for layer_idx in range(num_layers):
            hs = trace['hidden_states'][layer_idx]
            attn = trace['attentions'][layer_idx]
            
            seq_len_hs = hs.shape[0]
            num_heads, seq_len_attn_rows, seq_len_attn_cols = attn.shape
            
            if seq_len_hs != seq_len_attn_rows or seq_len_hs != seq_len_attn_cols:
                critical_issues.append(
                    f"   ❌ Capa {layer_idx}: hidden_states={seq_len_hs} tokens, "
                    f"attentions={seq_len_attn_rows}x{seq_len_attn_cols}"
                )
            elif layer_idx == 0:
                # Mostrar la primera capa como referencia
                print(f"   ✓ Capa {layer_idx}: {seq_len_hs} tokens consistentes "
                      f"(hidden_states y attentions coinciden)")
        
        if critical_issues:
            print(f"\n   ⚠️  PROBLEMAS ENCONTRADOS:")
            for issue in critical_issues[:5]:
                print(issue)
            if len(critical_issues) > 5:
                print(f"   ... y {len(critical_issues) - 5} capas más con problemas")
        else:
            print(f"   ✅ TODAS LAS CAPAS tienen dimensiones consistentes!")
        
        # Análisis de prompt vs respuesta
        if 'tokens_decoded' in trace:
            tokens_dec = trace['tokens_decoded']
            print(f"\n📝 ANÁLISIS PROMPT vs RESPUESTA:")
            
            # Buscar el separador típico "Answer:" o "A:"
            answer_markers = ['Answer:', 'A:', 'answer:', 'a:']
            prompt_length_estimated = None
            
            for i, token in enumerate(tokens_dec):
                for marker in answer_markers:
                    if marker in token:
                        prompt_length_estimated = i + 1
                        break
                if prompt_length_estimated:
                    break
            
            if prompt_length_estimated:
                response_length = len(tokens_dec) - prompt_length_estimated
                print(f"   Prompt estimado: ~{prompt_length_estimated} tokens")
                print(f"   Respuesta estimada: ~{response_length} tokens")
                print(f"   Total: {len(tokens_dec)} tokens")
            else:
                print(f"   No se encontró marcador de separación")
                print(f"   Total de tokens: {len(tokens_dec)}")
        
        print()
    
    # Resumen general
    print("="*80)
    print("RESUMEN")
    print("="*80)
    
    all_seq_lens_hs = []
    all_seq_lens_attn = []
    
    for _, _, trace in all_samples:
        for layer_idx in range(len(trace['hidden_states'])):
            hs = trace['hidden_states'][layer_idx]
            attn = trace['attentions'][layer_idx]
            
            all_seq_lens_hs.append(hs.shape[0])
            all_seq_lens_attn.append(attn.shape[1])
    
    print(f"\nEstadísticas de longitud de secuencia:")
    print(f"  hidden_states: min={min(all_seq_lens_hs)}, "
          f"max={max(all_seq_lens_hs)}, "
          f"promedio={np.mean(all_seq_lens_hs):.1f}")
    print(f"  attentions:    min={min(all_seq_lens_attn)}, "
          f"max={max(all_seq_lens_attn)}, "
          f"promedio={np.mean(all_seq_lens_attn):.1f}")
    
    if min(all_seq_lens_hs) == 1:
        print(f"\n⚠️  ADVERTENCIA: Hay secuencias de solo 1 token!")
        print(f"   Esto es inusual si se supone que incluye prompt + respuesta.")
        print(f"   Posibles causas:")
        print(f"   1. Solo se guardó la respuesta, no el prompt")
        print(f"   2. Hay un bug en la extracción")
        print(f"   3. Algunas respuestas son realmente de 1 token")
    
    if max(all_seq_lens_attn) > max(all_seq_lens_hs):
        print(f"\n❌ PROBLEMA CRÍTICO:")
        print(f"   Las atenciones tienen mayor dimensión que hidden_states!")
        print(f"   Esto causará el error 'index out of range'")
        print(f"   Solución: El dataloader ya recorta automáticamente las atenciones.")
    
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Inspeccionar estructura de traces"
    )
    
    parser.add_argument('--data-pattern', type=str, required=True,
                       help='Patrón glob para archivos .pkl')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Número de muestras a inspeccionar')
    
    args = parser.parse_args()
    
    inspect_trace_structure(args.data_pattern, args.num_samples)
