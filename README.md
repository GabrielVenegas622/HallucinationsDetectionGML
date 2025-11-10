<div style="display: flex; justify-content: space-around; align-items: center; width: 100%;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Escudo_de_la_Pontificia_Universidad_Cat%C3%B3lica_de_Chile.svg" width="72" alt="PUC Logo">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/47/Logo_UTFSM.png" width="100" alt="UTFSM Logo">
</div>

-----

# Hallucination Detection in LLM's with GML

This repository contains the source code for the _Graph Machine Learning (IIC3675)_ project by the lecturer Marcelo Mendoza (PUC) and authors Nicolás Schiaffino & Gabriel Venegas (UTFSM).

# Graphical Abstract

# 📁 Repository Structure

```
HallucinationsDetectionGML/
├── src/
│   ├── trace_extractor.py       # Extracts LLM traces (hidden states + attentions)
│   ├── trace_to_gt.py           # Generates ground truth scores using SelfCheckGPT
│   ├── dataloader.py            # Converts traces to graph sequences for training
│   └── baseline.py              # Trains GNN-det+LSTM for hallucination detection
├── traces_data/                 # Generated trace files (.pkl.gz)
├── notebooks/                   # EDA and prototyping notebooks
├── docs/                        # Additional documentation
├── ignore/                      # Development notes and old docs
└── README.md
```

# 📋 Pipeline de Ejecución

## 1️⃣ Extracción de Traces
Extrae hidden states y attentions del modelo LLM sobre el dataset.

```bash
python src/trace_extractor.py \
    --model-id llama2_chat_7B \
    --dataset triviaqa \
    --num-samples 1000 \
    --batch-size 50
```

**Salida:** `traces_data/*.pkl.gz` (archivos comprimidos con traces)

## 2️⃣ Generación de Ground Truth
Calcula scores de hallucination usando SelfCheckGPT.

```bash
python src/trace_to_gt.py \
    --traces-dir traces_data \
    --output-file ground_truth_scores.csv \
    --num-samples 10
```

**Salida:** `ground_truth_scores.csv` con scores [0,1] por respuesta

## 3️⃣ Entrenamiento del Modelo
Entrena GNN-det+LSTM para detección de alucinaciones.

```bash
python src/baseline.py \
    --data-pattern "traces_data/*.pkl*" \
    --scores-file ground_truth_scores.csv \
    --batch-size 16 \
    --epochs 50 \
    --score-threshold 0.5
```

**Salida:** Modelo entrenado con métricas AUROC

---

# ✅ ToDo

### Avance
- [x] Implementar **Llama2-7B-Chat** y generar respuestas sobre **TriviaQA**
- [x] Crear `trace_extractor.py` para extraer hidden states y attentions
- [x] Procesar traces y guardar con compresión gzip + float16
- [x] Implementar `dataloader.py` para convertir traces a grafos
- [x] Implementar `trace_to_gt.py` para generar ground truth con SelfCheckGPT
- [x] Implementar `baseline.py` con GNN-det+LSTM (metodología HaloScope)

### Entrega Final
- [ ] Entrenar modelo completo sobre dataset completo
- [ ] Implementar comparación con HaloScope original
- [ ] Ejecutar evaluación y generar gráficos comparativos
- [ ] Documentar resultados finales

### Propuesta 
- [x] Graphical abstract
- [x] Problema que se aborda en el proyecto
- [x] Técnicas a utilizar
- [x] Datos con los que se va a trabajar
- [x] Elementos Diferenciadores
- [x] Plan de actividades, Entregables
- [x] Video de 3 minutos

---

# 📚 Scripts Disponibles

## `src/trace_extractor.py`
Extrae hidden states y attention matrices de un modelo LLM.

**Uso:**
```bash
python src/trace_extractor.py \
    --model-id llama2_chat_7B \
    --dataset triviaqa \
    --num-samples 1000 \
    --batch-size 50
```

**Parámetros:**
- `--model-id`: Modelo a usar (llama2_chat_7B, qwen2.5_3B, etc.)
- `--dataset`: Dataset (triviaqa, truthfulqa)
- `--num-samples`: Número de muestras a procesar
- `--batch-size`: Muestras por archivo (default: 50)

**Salida:** `traces_data/*.pkl.gz` con hidden_states + attentions (float16 comprimido)

---

## `src/trace_to_gt.py`
Genera ground truth scores usando SelfCheckGPT.

**Uso:**
```bash
python src/trace_to_gt.py \
    --traces-dir traces_data \
    --output-file ground_truth_scores.csv \
    --num-samples 10
```

**Parámetros:**
- `--traces-dir`: Directorio con traces (.pkl o .pkl.gz)
- `--output-file`: Archivo CSV de salida
- `--num-samples`: Número de samples por pregunta (default: 10)
- `--batch-size`: Batch size para procesamiento (default: 4)

**Salida:** CSV con columnas `[question_id, hallucination_score]`

---

## `src/dataloader.py`
Carga traces y los convierte en grafos para PyTorch Geometric.

**Uso en código:**
```python
from dataloader import HallucinationDataset

dataset = HallucinationDataset(
    data_pattern="traces_data/*.pkl*",
    scores_file="ground_truth_scores.csv",
    score_threshold=0.5
)

# DataLoader para entrenamiento
from torch_geometric.loader import DataLoader
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

**Estructura del grafo:**
- Nodos: Tokens de la respuesta
- Features: Hidden states de última capa
- Aristas: Atención promediada entre tokens
- Label: 0 (no alucinado) o 1 (alucinado)

---

## `src/baseline.py`
Entrena modelo GNN-det+LSTM basado en metodología HaloScope.

**Uso:**
```bash
python src/baseline.py \
    --data-pattern "traces_data/*.pkl*" \
    --scores-file ground_truth_scores.csv \
    --batch-size 16 \
    --epochs 50 \
    --score-threshold 0.5 \
    --learning-rate 0.001
```

**Parámetros principales:**
- `--data-pattern`: Patrón de archivos de traces
- `--scores-file`: CSV con ground truth scores
- `--score-threshold`: Umbral para clasificar alucinación (default: 0.5)
- `--batch-size`: Tamaño de batch (default: 16)
- `--epochs`: Épocas de entrenamiento (default: 50)

**Arquitectura:**
- **GNN-det:** 3 capas GINE para procesar grafos por capa
- **LSTM:** Procesa secuencia de embeddings de capas
- **Loss:** Binary Cross Entropy
- **Métrica:** AUROC

**Salida:** Modelo entrenado + métricas de evaluación

---
