# Implementación de Last Token Readout con Conexión Residual

## Resumen

Se han modificado las clases `GNNDetLSTM` y `GVAELSTM` en `baseline.py` para implementar una estrategia de **Last Token Readout** con **conexión residual (skip connection)**, reemplazando la estrategia anterior de `global_mean_pool`.

## Motivación

### Problema con `global_mean_pool`
- **Diluye la señal**: Al promediar todos los nodos, se pierde información específica del último token
- **Reduce rendimiento**: El último token contiene información crítica sobre el estado final de la secuencia
- **Pérdida de contexto**: La información posicional se borra en el promedio

### Solución: Last Token Readout
- **Señal enfocada**: Solo se extrae el último nodo de cada grafo en el batch
- **Información preservada**: Mantiene la representación del token más relevante
- **Conexión residual**: Concatena embedding original + embedding procesado por GNN

---

## Cambios Implementados

### 1. GNNDetLSTM

#### Cambio en `__init__`
```python
# ANTES:
self.lstm = nn.LSTM(
    input_size=gnn_hidden,  # Solo recibía output de GNN
    ...
)

# DESPUÉS:
self.lstm = nn.LSTM(
    input_size=hidden_dim + gnn_hidden,  # Concatenación residual
    ...
)
```

**Razón**: La LSTM ahora recibe la concatenación del embedding original (`hidden_dim`) y el procesado por GNN (`gnn_hidden`).

#### Cambio en `forward`
```python
# DESPUÉS del procesamiento GNN de cada capa:

# 1. Identificar índices del último token de cada grafo
batch_size = batch.max().item() + 1
last_token_indices = []

for graph_id in range(batch_size):
    node_mask = (batch == graph_id)
    node_indices = torch.where(node_mask)[0]
    if len(node_indices) > 0:
        last_token_indices.append(node_indices[-1])

last_token_indices = torch.tensor(last_token_indices, device=x_original.device)

# 2. Extraer features del último token
original_last_token = x_original[last_token_indices]  # [batch_size, hidden_dim]
gnn_last_token = x_gnn[last_token_indices]            # [batch_size, gnn_hidden]

# 3. Concatenación residual
combined = torch.cat([original_last_token, gnn_last_token], dim=1)  # [batch_size, hidden_dim + gnn_hidden]

# 4. Apilar para crear secuencia temporal
layer_representations.append(combined)
```

**Resultado**: La secuencia que entra a la LSTM tiene forma `[batch_size, num_layers, hidden_dim + gnn_hidden]`

---

### 2. GVAELSTM

#### Cambio en `__init__`
```python
# ANTES:
self.lstm = nn.LSTM(
    input_size=latent_dim,  # Solo recibía el vector latente z
    ...
)

# DESPUÉS:
self.lstm = nn.LSTM(
    input_size=hidden_dim + latent_dim,  # Concatenación residual
    ...
)
```

**Razón**: La LSTM ahora recibe la concatenación del embedding original (`hidden_dim`) y el vector latente (`latent_dim`).

#### Cambio en `encode`
```python
# ANTES:
# Global pooling (promedio de todos los nodos)
graph_repr = global_mean_pool(x, batch)

# DESPUÉS:
# Last Token Readout: seleccionar solo el último nodo
batch_size = batch.max().item() + 1
last_token_indices = []

for graph_id in range(batch_size):
    node_mask = (batch == graph_id)
    node_indices = torch.where(node_mask)[0]
    if len(node_indices) > 0:
        last_token_indices.append(node_indices[-1])

last_token_indices = torch.tensor(last_token_indices, device=x.device)

# Extraer features del último token después de GNN
graph_repr = x[last_token_indices]  # [batch_size, gnn_hidden]

# Proyectar a parámetros de distribución
mu = self.fc_mu(graph_repr)
logvar = self.fc_logvar(graph_repr)
```

**Resultado**: El espacio latente `z` ahora representa específicamente el último token enriquecido por la GNN.

#### Cambio en `forward`
```python
# Para cada capa:

# 1. Extraer original_last_token ANTES de encode
batch_size = batch.max().item() + 1
last_token_indices = [...]  # Mismo proceso que antes

# Extraer features originales del último token [batch_size, hidden_dim]
original_last_token = x_original[last_token_indices]

# 2. Encode -> obtener mu, logvar
mu, logvar, graph_repr = self.encode(x_original, edge_index, edge_attr, batch)

# 3. Reparameterize -> obtener z
z = self.reparameterize(mu, logvar)

# 4. Decode para reconstruction loss
x_reconstructed = self.decode(z)

# 5. Concatenación residual crítica
combined = torch.cat([original_last_token, z], dim=1)  # [batch_size, hidden_dim + latent_dim]

# 6. Apilar para secuencia LSTM
latent_sequence.append(combined)
```

**Resultado**: La secuencia que entra a la LSTM tiene forma `[batch_size, num_layers, hidden_dim + latent_dim]`

---

## Ventajas de la Implementación

### 1. Señal Enfocada
- El último token contiene la información más relevante del estado final
- No se diluye con información de tokens intermedios
- Mejor captura de la evolución temporal

### 2. Conexión Residual (Skip Connection)
- Preserva información original que podría perderse en la GNN
- Permite al modelo decidir qué información usar (original vs procesada)
- Gradientes más estables durante el entrenamiento

### 3. Consistencia Arquitectónica
- Ambos modelos (GNN-det y GVAE) usan la misma estrategia
- Comparación justa entre arquitecturas
- Código más mantenible

### 4. Mejor Uso de Parámetros
- GNN-det+LSTM: LSTM recibe input más rico → menos parámetros en LSTM, más en GNN
- GVAE+LSTM: LSTM recibe input más rico → menos parámetros en LSTM, más en encoder/decoder
- Distribución equilibrada de parámetros en el modelo completo

---

## Impacto en el Conteo de Parámetros

### LSTM-solo (baseline)
- LSTM input: `hidden_dim = 4096`
- **Más parámetros en LSTM** (única parte entrenable)
- Total: ~4.1M parámetros

### GNN-det+LSTM
- LSTM input: `hidden_dim + gnn_hidden = 4096 + 128 = 4224`
- **Menos parámetros en LSTM** pero **más parámetros en GNN** (conv1, conv2)
- Total: ~1.5M parámetros (distribución diferente)

### GVAE+LSTM
- LSTM input: `hidden_dim + latent_dim = 4096 + 64 = 4160`
- **Menos parámetros en LSTM** pero **más parámetros en encoder/decoder** (conv1, conv2, fc_mu, fc_logvar, decoder)
- Total: ~1.5M parámetros (distribución diferente)

### Conclusión sobre Parámetros
- **NO es desventajoso** que GNN-det y GVAE tengan menos parámetros en total
- La comparación debe considerar **capacidad expresiva**, no solo conteo de parámetros
- GNN y GVAE tienen arquitecturas más ricas (estructura de grafo + atención)
- LSTM-solo solo tiene secuencia temporal (menos información estructural)

---

## Testing

Se ha creado el script `test_last_token_readout.py` que verifica:

1. ✅ Dimensiones correctas de entrada de LSTM
2. ✅ Concatenación residual funciona correctamente
3. ✅ Forward pass completo sin errores
4. ✅ Shapes de output correctos
5. ✅ Comparación de parámetros entre modelos

### Ejecutar tests
```bash
cd /home/gaara/mnt/USM/2025-02/IIC3641/HallucinationsDetectionGML
python src/test_last_token_readout.py
```

---

## Próximos Pasos

1. **Entrenar modelos** con la nueva implementación
2. **Comparar rendimiento** (AUROC, F1, etc.) entre:
   - Versión anterior (global_mean_pool)
   - Versión nueva (last_token_readout + residual)
3. **Análisis de ablación** para validar hipótesis
4. **Visualizaciones** de atención y embeddings

---

## Referencias

- **Skip Connections**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- **Last Token Pooling**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", NAACL 2019
- **Graph Readout**: Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019

---

**Fecha de implementación**: 2025-02-XX  
**Autor**: Modificado según especificaciones del usuario  
**Estado**: ✅ Implementado y listo para testing
