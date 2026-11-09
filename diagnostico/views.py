import base64
import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import clasificador

EXTENSIONES = {"jpg", "jpeg", "png", "webp"}


def _validar(archivo):
    if archivo is None:
        return "Selecciona una imagen de una hoja de café."
    ext = archivo.name.rsplit(".", 1)[-1].lower() if "." in archivo.name else ""
    if ext not in EXTENSIONES:
        return "Formato no válido. Usa JPG, PNG o WEBP."
    return None


def inicio(request):
    contexto = {"metricas": clasificador.metricas()}
    if request.method == "POST":
        archivo = request.FILES.get("imagen")
        error = _validar(archivo)
        if error:
            contexto["error"] = error
        else:
            try:
                datos = archivo.read()
                archivo.seek(0)
                resultado = clasificador.predecir(archivo)
            except Exception:
                contexto["error"] = "No se pudo leer la imagen. Intenta con otro archivo."
            else:
                tipo = archivo.content_type or "image/jpeg"
                contexto["resultado"] = resultado
                contexto["resultado_json"] = json.dumps(resultado["todas"])
                contexto["imagen_b64"] = f"data:{tipo};base64,{base64.b64encode(datos).decode()}"
    return render(request, "diagnostico/inicio.html", contexto)


def resultados(request):
    m = clasificador.metricas()
    return render(request, "diagnostico/resultados.html", {"m": m, "m_json": json.dumps(m)})


def acerca(request):
    return render(request, "diagnostico/acerca.html", {"m": clasificador.metricas()})


@csrf_exempt
@require_POST
def api_predecir(request):
    archivo = request.FILES.get("imagen")
    error = _validar(archivo)
    if error:
        return JsonResponse({"error": error}, status=400)
    try:
        return JsonResponse(clasificador.predecir(archivo))
    except Exception:
        return JsonResponse({"error": "No se pudo procesar la imagen."}, status=400)
