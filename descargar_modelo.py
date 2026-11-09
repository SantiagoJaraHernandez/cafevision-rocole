"""Verifica que el modelo TFLite esté en modelo/ durante el build en Render.

Si el archivo no está en el repositorio, lo descarga de un enlace público de
Google Drive indicado en la variable de entorno MODELO_DRIVE_ID.
"""
import os
import sys
import urllib.request
from pathlib import Path

destino = Path(__file__).parent / "modelo" / "modelo_cafe.tflite"
if destino.exists() and destino.stat().st_size > 100_000:
    print("Modelo ya presente:", destino)
    sys.exit(0)
id_ = os.environ.get("MODELO_DRIVE_ID")
if not id_:
    sys.exit("Falta la variable MODELO_DRIVE_ID y no existe modelo/modelo_cafe.tflite")
url = f"https://drive.usercontent.google.com/download?id={id_}&export=download&confirm=t"
destino.parent.mkdir(exist_ok=True)
urllib.request.urlretrieve(url, destino)
datos = destino.read_bytes()
if len(datos) < 100_000 or datos[4:8] != b"TFL3":
    destino.unlink()
    sys.exit("La descarga no es un modelo TFLite válido (¿el archivo es público?)")
print(f"Modelo descargado: {len(datos) / 1e6:.2f} MB")
