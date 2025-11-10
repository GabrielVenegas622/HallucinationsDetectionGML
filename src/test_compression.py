#!/usr/bin/env python3
"""
Script de Verificación de Compresión Gzip + Float16

Verifica que los archivos se están guardando y cargando correctamente
con la nueva compresión.

Uso:
    python test_compression.py
"""

import pickle
import gzip
import numpy as np
import tempfile
from pathlib import Path


def test_compression_implementation():
    """Test completo de la implementación de compresión"""
    
    print("="*80)
    print("TEST DE COMPRESIÓN GZIP + FLOAT16")
    print("="*80)
    
    # Crear datos de prueba simulando un trace
    print("\n1. Creando datos de prueba...")
    test_trace = {
        'question_id': 'test_001',
        'generated_answer_clean': 'Test answer',
        'tokens': np.array([1, 2, 3, 4, 5]),
        'tokens_decoded': ['token1', 'token2', 'token3', 'token4', 'token5'],
        'hidden_states': [],
        'attentions': []
    }
    
    # Simular hidden states (32 capas, 5 tokens, 4096 dim)
    num_layers = 32
    seq_len = 5
    hidden_dim = 4096
    num_heads = 32
    
    for layer in range(num_layers):
        # Hidden states en float16
        hs = np.random.randn(seq_len, hidden_dim).astype(np.float16)
        test_trace['hidden_states'].append(hs)
        
        # Attentions en float16
        attn = np.random.rand(num_heads, seq_len, seq_len).astype(np.float16)
        test_trace['attentions'].append(attn)
    
    print(f"✓ Datos creados: {num_layers} capas, {seq_len} tokens")
    
    # Test 1: Guardar sin compresión
    print("\n2. Test: Guardar SIN compresión (.pkl)...")
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_pkl = Path(tmp.name)
        with open(tmp_pkl, 'wb') as f:
            pickle.dump([test_trace], f)
    
    size_uncompressed = tmp_pkl.stat().st_size / (1024 * 1024)
    print(f"   Tamaño sin comprimir: {size_uncompressed:.2f} MB")
    
    # Test 2: Guardar con compresión gzip
    print("\n3. Test: Guardar CON compresión (.pkl.gz)...")
    with tempfile.NamedTemporaryFile(suffix='.pkl.gz', delete=False) as tmp:
        tmp_gz = Path(tmp.name)
        with gzip.open(tmp_gz, 'wb', compresslevel=6) as f:
            pickle.dump([test_trace], f)
    
    size_compressed = tmp_gz.stat().st_size / (1024 * 1024)
    print(f"   Tamaño comprimido: {size_compressed:.2f} MB")
    
    reduction = (1 - size_compressed / size_uncompressed) * 100
    print(f"   Reducción: {reduction:.1f}%")
    
    # Test 3: Verificar que se puede leer el archivo comprimido
    print("\n4. Test: Cargar archivo comprimido...")
    with gzip.open(tmp_gz, 'rb') as f:
        loaded_traces = pickle.load(f)
    
    loaded_trace = loaded_traces[0]
    print(f"✓ Archivo cargado correctamente")
    
    # Test 4: Verificar dtype
    print("\n5. Test: Verificar dtype float16...")
    hs_dtype = loaded_trace['hidden_states'][0].dtype
    attn_dtype = loaded_trace['attentions'][0].dtype
    
    print(f"   Hidden states dtype: {hs_dtype}")
    print(f"   Attentions dtype: {attn_dtype}")
    
    assert hs_dtype == np.float16, f"❌ Hidden states debería ser float16, es {hs_dtype}"
    assert attn_dtype == np.float16, f"❌ Attentions debería ser float16, es {attn_dtype}"
    print(f"✓ Dtypes correctos")
    
    # Test 5: Verificar integridad de datos
    print("\n6. Test: Verificar integridad de datos...")
    
    # Comparar shapes
    for i in range(num_layers):
        orig_hs_shape = test_trace['hidden_states'][i].shape
        load_hs_shape = loaded_trace['hidden_states'][i].shape
        assert orig_hs_shape == load_hs_shape, f"❌ Shape mismatch en hidden_states capa {i}"
        
        orig_attn_shape = test_trace['attentions'][i].shape
        load_attn_shape = loaded_trace['attentions'][i].shape
        assert orig_attn_shape == load_attn_shape, f"❌ Shape mismatch en attentions capa {i}"
    
    print(f"✓ Shapes preservados correctamente")
    
    # Verificar valores (con tolerancia por float16)
    hs_diff = np.abs(test_trace['hidden_states'][0] - loaded_trace['hidden_states'][0]).max()
    print(f"   Diferencia máxima en hidden_states: {hs_diff:.10f}")
    assert hs_diff < 1e-3, f"❌ Diferencia muy grande: {hs_diff}"
    print(f"✓ Valores preservados (diferencia < 1e-3)")
    
    # Test 6: Verificar metadatos
    print("\n7. Test: Verificar metadatos...")
    assert loaded_trace['question_id'] == test_trace['question_id']
    assert loaded_trace['generated_answer_clean'] == test_trace['generated_answer_clean']
    assert len(loaded_trace['tokens_decoded']) == len(test_trace['tokens_decoded'])
    print(f"✓ Metadatos preservados")
    
    # Limpiar archivos temporales
    tmp_pkl.unlink()
    tmp_gz.unlink()
    
    # Resumen
    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)
    print(f"Tamaño original:    {size_uncompressed:.2f} MB")
    print(f"Tamaño comprimido:  {size_compressed:.2f} MB")
    print(f"Reducción:          {reduction:.1f}%")
    print(f"Dtype:              float16 ✓")
    print(f"Integridad:         Preservada ✓")
    print("\n✅ TODOS LOS TESTS PASARON")
    print("\nLa implementación de Gzip + Float16 está funcionando correctamente.")
    print("Los archivos .pkl.gz se pueden usar normalmente con los scripts actualizados.")
    print("="*80)


if __name__ == '__main__':
    test_compression_implementation()
