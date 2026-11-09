"""Carga el modelo TFLite (MobileNetV2 con fine-tuning) y hace la predicción.

El modelo incluye la capa Rescaling(1/127.5, offset=-1), así que recibe la
imagen RGB de 224x224 con píxeles en el rango 0-255, igual que en el
entrenamiento con image_dataset_from_directory.
"""
import json
import threading
from functools import lru_cache

import numpy as np
from django.conf import settings
from PIL import Image, ImageOps

TAMANO = (224, 224)

INFO_CLASES = {
    "coffee___healthy": {
        "nombre": "Hoja sana",
        "color": "#2e7d32",
        "descripcion": "No se observan síntomas de roya ni de ácaro rojo.",
        "recomendacion": "Mantener el monitoreo periódico del cultivo y las prácticas de manejo actuales.",
    },
    "coffee___rust": {
        "nombre": "Roya del café",
        "color": "#e65100",
        "descripcion": "Enfermedad causada por el hongo Hemileia vastatrix; produce manchas "
                       "amarillo-anaranjadas en el envés de la hoja.",
        "recomendacion": "Confirmar con un agrónomo, revisar las plantas vecinas y evaluar el "
                         "control según el nivel de incidencia en el lote.",
    },
    "coffee___red_spider_mite": {
        "nombre": "Ácaro rojo",
        "color": "#c62828",
        "descripcion": "Plaga (Oligonychus spp.) que causa un bronceado o enrojecimiento de "
                       "la hoja por la alimentación del ácaro.",
        "recomendacion": "Confirmar con un agrónomo e inspeccionar el envés de las hojas con lupa; "
                         "es más frecuente en épocas secas.",
    },
}

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _cargar():
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # entorno con TensorFlow completo
        from tensorflow.lite.python.interpreter import Interpreter
    interp = Interpreter(model_path=str(settings.MODEL_DIR / "modelo_cafe.tflite"))
    interp.allocate_tensors()
    clases = json.loads((settings.MODEL_DIR / "class_names.json").read_text())
    return interp, clases


def preprocesar(archivo) -> np.ndarray:
    img = Image.open(archivo)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize(TAMANO, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)[None, ...]


def predecir(archivo) -> dict:
    x = preprocesar(archivo)
    interp, clases = _cargar()
    entrada = interp.get_input_details()[0]
    salida = interp.get_output_details()[0]
    with _lock:  # el intérprete TFLite no es seguro entre hilos
        interp.set_tensor(entrada["index"], x)
        interp.invoke()
        probs = interp.get_tensor(salida["index"])[0].astype(float)
    orden = np.argsort(probs)[::-1]
    resultados = [
        {"clase": clases[i], "probabilidad": round(float(probs[i]) * 100, 2), **INFO_CLASES[clases[i]]}
        for i in orden
    ]
    return {"prediccion": resultados[0], "todas": resultados}


@lru_cache(maxsize=1)
def metricas() -> dict:
    return json.loads((settings.MODEL_DIR / "metricas.json").read_text())
