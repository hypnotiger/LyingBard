from django.urls import path, register_converter

from . import views, converters

register_converter(converters.B64IDConverter, "b64id")

app_name = "lyingbard"
urlpatterns = [
    path("", views.index, name="index"),
    path("generate/", views.generate, name="generate"),
    path("generate/widget/", views.generate_widget, name="generate_widget"),
    path("dashboard", views.dashboard, name="dashboard"),
    path("speakers/<b64id:speaker_id>/", views.speaker_details, name="speaker_details"),
    path("generations/", views.generations, name="generations"),
    path("generations/delete/", views.generations_delete, name="generations_delete"),
    path("generations/widget/", views.generations_widget, name="generations_widget"),
    path("generations/<b64id:generation_id>/", views.generation_details, name="generation_details"),
    path("generations/<b64id:generation_id>/audio", views.generation_audio, name="generation_audio"),
]