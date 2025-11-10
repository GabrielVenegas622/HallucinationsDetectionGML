#!/usr/bin/env python3
"""
Script de Validación de Traces

Verifica que los datos en .pkl no tengan inconsistencias entre
hidden_states y attentions que causen errores de índices.

Uso:
    python validate_traces.py --data-pattern "traces_data/*.pkl"
"""

import pickle
import glob
import numpy as np
import argparse
from tqdm import tqdm


def validate_trace(trace, trace_idx, file_path):
    """Valida un trace individual"""
    issues = []
    
    question_id = trace.get('question_id', 'unknown')
    
    # Validar que existan las claves necesarias
    if 'hidden_states' not in trace:
        issues.append(f"Trace {trace_idx} ({question_id}): Falta 'hidden_states'")
        return issues
    
    if 'attentions' not in trace:
        issues.append(f"Trace {trace_idx} ({question_id}): Falta 'attentions'")
        return issues
    
    num_layers = len(trace['hidden_states'])
    
    # Validar cada capa
    for layer_idx in range(num_layers):
        hs = trace['hidden_states'][layer_idx]
        attn = trace['attentions'][layer_idx]
        
        # Validar hidden_states
        if not isinstance(hs, np.ndarray):
            issues.append(f"Trace {trace_idx}, capa {layer_idx}: hidden_states no es numpy array")
            continue
        
        if len(hs.shape) != 2:
            issues.append(f"Trace {trace_idx}, capa {layer_idx}: hidden_states tiene shape {hs.shape}, esperado (seq_len, hidden_dim)")
            continue
        
        seq_len, hidden_dim = hs.shape
        
        # Validar attentions
        if not isinstance(attn, np.ndarray):
            issues.append(f"Trace {trace_idx}, capa {layer_idx}: attentions no es numpy array")
            continue
        
        if len(attn.shape) != 3:
            issues.append(f"Trace {trace_idx}, capa {layer_idx}: attentions tiene shape {attn.shape}, esperado (num_heads, seq_len, seq_len)")
            continue
        
        num_heads, attn_rows, attn_cols = attn.shape
        
        # VALIDACIÓN CRÍTICA: attn debe ser [num_heads, seq_len, seq_len]
        if attn_rows != seq_len or attn_cols != seq_len:
            issues.append(
                f"❌ CRÍTICO - Trace {trace_idx} ({question_id}), capa {layer_idx}: "
                f"Mismatch entre hidden_states y attentions\n"
                f"   hidden_states shape: ({seq_len}, {hidden_dim})\n"
                f"   attentions shape: ({num_heads}, {attn_rows}, {attn_cols})\n"
                f"   ⚠️  Los índices de atención ({attn_rows}x{attn_cols}) no coinciden con seq_len ({seq_len})\n"
                f"   Esto causará 'index out of range' al crear grafos!"
            )
        
        # Validar valores NaN/Inf
        if np.isnan(hs).any():
            issues.append(f"⚠️  Trace {trace_idx}, capa {layer_idx}: NaN en hidden_states")
        
        if np.isinf(hs).any():
            issues.append(f"⚠️  Trace {trace_idx}, capa {layer_idx}: Inf en hidden_states")
        
        if np.isnan(attn).any():
            issues.append(f"⚠️  Trace {trace_idx}, capa {layer_idx}: NaN en attentions")
        
        if np.isinf(attn).any():
            issues.append(f"⚠️  Trace {trace_idx}, capa {layer_idx}: Inf en attentions")
        
        # Validar rango de valores en attentions [0, 1]
        if attn.min() < 0 or attn.max() > 1:
            issues.append(
                f"⚠️  Trace {trace_idx}, capa {layer_idx}: "
                f"attentions fuera de rango [0,1]: [{attn.min():.4f}, {attn.max():.4f}]"
            )
    
    return issues


def validate_all_traces(file_pattern):
    """Valida todos los traces en los archivos .pkl"""
    print("="*80)
    print("VALIDACIÓN DE TRACES")
    print("="*80)
    
    files = glob.glob(file_pattern)
    print(f"\nEncontrados {len(files)} archivos .pkl")
    
    if len(files) == 0:
        print(f"❌ No se encontraron archivos con el patrón: {file_pattern}")
        return
    
    all_issues = []
    total_traces = 0
    traces_with_issues = 0
    critical_issues = 0
    
    for file_path in tqdm(files, desc="Validando archivos"):
        try:
            with open(file_path, 'rb') as f:
                traces = pickle.load(f)
            
            for trace_idx, trace in enumerate(traces):
                total_traces += 1
                issues = validate_trace(trace, trace_idx, file_path)
                
                if issues:
                    traces_with_issues += 1
                    for issue in issues:
                        all_issues.append(f"Archivo: {file_path}\n  {issue}")
                        if "CRÍTICO" in issue:
                            critical_issues += 1
        
        except Exception as e:
            all_issues.append(f"❌ ERROR al cargar {file_path}: {e}")
    
    # Reporte
    print("\n" + "="*80)
    print("RESULTADOS DE VALIDACIÓN")
    print("="*80)
    print(f"\nTotal de traces analizados: {total_traces}")
    print(f"Traces con problemas: {traces_with_issues}")
    print(f"Problemas críticos encontrados: {critical_issues}")
    
    if all_issues:
        print(f"\n⚠️  Se encontraron {len(all_issues)} problemas:\n")
        
        # Mostrar primeros 20 problemas
        for i, issue in enumerate(all_issues[:20]):
            print(issue)
            print()
        
        if len(all_issues) > 20:
            print(f"... y {len(all_issues) - 20} problemas más\n")
        
        # Recomendaciones
        print("="*80)
        print("RECOMENDACIONES")
        print("="*80)
        
        if critical_issues > 0:
            print("\n❌ HAY PROBLEMAS CRÍTICOS que causarán errores de índices!")
            print("\nPosibles causas:")
            print("  1. Las atenciones fueron extraídas con una seq_len diferente a hidden_states")
            print("  2. Hay padding en las atenciones que no coincide con los tokens reales")
            print("  3. Se usó un modelo con cache activado que modifica las dimensiones")
            print("\nSoluciones:")
            print("  a) Re-extraer los traces asegurando que attn.shape[1:] == hidden_states.shape[0]")
            print("  b) Usar el dataloader actualizado que recorta automáticamente")
            print("  c) Limpiar los datos manualmente con el script de normalización")
        else:
            print("\n✓ No hay problemas críticos")
            if all_issues:
                print("  Solo hay warnings (NaN, Inf, valores fuera de rango)")
                print("  El código los manejará automáticamente")
    else:
        print("\n✅ ¡TODOS LOS TRACES SON VÁLIDOS!")
        print("   No se encontraron problemas.")
    
    print("="*80)
    
    return critical_issues == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Validar traces en archivos .pkl"
    )
    
    parser.add_argument('--data-pattern', type=str, required=True,
                       help='Patrón glob para archivos .pkl (ej: "traces_data/*.pkl")')
    
    args = parser.parse_args()
    
    success = validate_all_traces(args.data_pattern)
    
    exit(0 if success else 1)
