from django.urls import path

from . import views

app_name = "diagnostico"

urlpatterns = [
    path("", views.inicio, name="inicio"),
    path("resultados/", views.resultados, name="resultados"),
    path("acerca/", views.acerca, name="acerca"),
    path("api/predecir/", views.api_predecir, name="api_predecir"),
]
