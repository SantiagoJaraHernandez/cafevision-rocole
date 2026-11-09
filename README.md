# CaféVision: diagnóstico de hojas de café robusta con redes neuronales

Proyecto final de **Inteligencia Artificial**, Ingeniería de Sistemas, CORHUILA (2026-2).
Docente: José Miguel Llanos Mosquera.

Aplicación web en **Django** que clasifica fotos de hojas de café robusta en tres clases
(**sana**, **roya** y **ácaro rojo**) con una red neuronal convolucional (MobileNetV2 con
*transfer learning*) entrenada con el dataset **RoCoLe**.

## Resultados

Conjunto de prueba: 260 imágenes de plantas que el modelo no vio en el entrenamiento.

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| Hoja sana | 82,8 % | 94,0 % | 88,0 % | 133 |
| Ácaro rojo | 45,7 % | 61,5 % | 52,5 % | 26 |
| Roya | 86,5 % | 63,4 % | 73,1 % | 101 |
| **Accuracy** | | | **78,8 %** | 260 |
| F1 macro | | | 71,2 % | |

| Modelo | Accuracy validación | Accuracy prueba |
|---|---|---|
| `modelo_cafe_v2.keras` | 73,8 % | 76,2 % |
| `modelo_cafe_v3.keras` (elegido por validación) | 75,8 % | 76,5 % |
| v3 → TFLite con cuantización dinámica (desplegado) | — | 78,8 % |

El modelo se eligió por su accuracy en validación. El conjunto de prueba solo se usó para el reporte final.

## Dataset

[RoCoLe: Robusta Coffee Leaf Images](https://www.kaggle.com/datasets/nirmalsankalana/rocole-a-robusta-coffee-leaf-images-dataset):
1.560 fotos tomadas en campo con smartphone. Hay 791 hojas sanas, 602 con roya y 167 con ácaro rojo.

## Metodología

1. **Partición por planta.** Se extrae el identificador de planta (`C#P#`) del nombre del archivo y se usa
   `StratifiedGroupKFold`, así que todas las fotos de una planta quedan en una sola partición.
   El resultado es 1.040 imágenes de entrenamiento, 260 de validación y 260 de prueba, sin fuga de datos entre particiones.
2. **Preprocesamiento.** Las imágenes se redimensionan a 224×224 y los píxeles se escalan a [-1, 1] con una capa `Rescaling` dentro del modelo.
3. **Aumento de datos.** Se aplican volteos, rotación, zoom, contraste y brillo aleatorios.
4. **Desbalance.** Se usan pesos de clase balanceados.
5. **Fase 1.** MobileNetV2 (preentrenada en ImageNet) va congelada y solo se entrena la cabeza (GAP + Dropout 0,3 + Dense softmax) con Adam 1e-3.
6. **Fase 2.** Fine-tuning de las últimas 30 capas con Adam 1e-5, EarlyStopping y ReduceLROnPlateau.
7. **Fase 3.** Ajuste fino adicional con Adam 5e-5. EarlyStopping (paciencia 8) restaura los pesos con menor `val_loss`.
8. **Exportación.** El modelo se convierte a TensorFlow Lite con cuantización dinámica (≈2,5 MB) para servirlo con poca memoria.

El entrenamiento está en el cuaderno de Google Colab `Training Proyecto IA.ipynb`.

## La aplicación

| Ruta | Contenido |
|---|---|
| `/` | Subir una foto y ver la clase predicha, la confianza y la probabilidad por clase |
| `/resultados/` | Curvas de accuracy y loss, matriz de confusión, métricas por clase y distribución de clases |
| `/acerca/` | Metodología |
| `POST /api/predecir/` | API JSON (campo `imagen`) |

```bash
curl -X POST -F "imagen=@hoja.jpg" https://<tu-app>.onrender.com/api/predecir/
```

### Estructura

```
config/                 configuración de Django (settings, urls, wsgi)
diagnostico/
  clasificador.py       carga del modelo TFLite, preprocesamiento y predicción
  views.py              vistas web y API
  templates/            páginas (Chart.js para las gráficas)
  static/               estilos
modelo/
  class_names.json      orden de las clases del modelo
  metricas.json         métricas de prueba e historial de entrenamiento (alimenta las gráficas)
  modelo_cafe.tflite    modelo entrenado (TensorFlow Lite, ≈2,5 MB)
descargar_modelo.py     alternativa: descarga el .tflite desde Drive si no está en el repo
construir_metricas.py   genera modelo/metricas.json a partir de la evaluación en Colab
build.sh / render.yaml  despliegue en Render
```

## Ejecutar en local

```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver
```

Abrir http://127.0.0.1:8000

## Despliegue en Render

1. En Render, ir a **New → Blueprint** y conectar este repositorio. Render lee `render.yaml`.
2. Variables de entorno: `SECRET_KEY` se genera sola y `DEBUG=0`. El modelo ya va en `modelo/`.
3. Build: `bash build.sh`. Start: `gunicorn config.wsgi:application`.

Se usa `ai-edge-litert` (el runtime de TensorFlow Lite) en lugar de TensorFlow completo, así que la app cabe en el plan gratuito de 512 MB.

## Limitaciones

- La clase **ácaro rojo** tiene pocas imágenes (167), por eso su precisión es baja (45,7 %): el modelo confunde con ella algunas hojas con roya.
- La **roya** se confunde a veces con hojas sanas (22 de 101 casos), sobre todo cuando las lesiones son incipientes.
- Es una herramienta académica y no reemplaza el diagnóstico de un agrónomo.

## Referencias

- Parraga-Alava, J., Cusme, K., Loor, A., & Santander, E. (2019). RoCoLe: A robusta coffee leaf images dataset for evaluation of machine learning based methods in plant diseases recognition. *Data in Brief, 25*, 104414. https://doi.org/10.1016/j.dib.2019.104414
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 4510–4520.
