<div style="display: flex; justify-content: space-around; align-items: center; width: 100%;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Escudo_de_la_Pontificia_Universidad_Cat%C3%B3lica_de_Chile.svg" width="72" alt="PUC Logo">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/47/Logo_UTFSM.png" width="100" alt="UTFSM Logo">
</div>

# Hallucination Detection in LLM's with GML

This repository contains the source code for the _Graph Machine Learning (IIC3675)_ project by the lecturer Marcelo Mendoza (PUC) and authors Nicolás Schiaffino & Gabriel Venegas (UTFSM).

# ToDo

### Avance
- [ ] Implementar Llama-1B y generar las respuestas de las preguntas de _TruthfulQA_
- [ ] Crear script para extraer las atenciones y embeddings de todas las capas para la respuesta generada.
- [ ] Procesar todas las respuestas con el script para recuperar atenciones y embeddings.
- [ ] Generar el dataset con cada fila como `[id_pregunta, respuesta, atenciones_capa_k, activaciones_capa_k, ..., atenciones_capa_N, activaciones_capa_N]`
- [ ] Implementar un proceso de carga de cada fila de dataset para generar el Grafo.
- [ ] Implementar VAE.

### Entrega Final

### Propuesta 
- [x] Graphical abstract.
- [x] Problema que se aborda en el proyecto.
- [x] Técnicas a utilizar.
- [x] Datos con los que se va a trabajar.
- [x] Elementos Diferenciadores
- [x] Plan de actividades, Entregables al avance y a la entrega final.
- [x] Video de 3 minutos.