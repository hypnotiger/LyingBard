from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("generate", views.generate, name="generate"),
    path("speakers/<int:speaker_id>/", views.speaker_details, name="speaker_details"),
    path("generations/<int:generation_id>/", views.generation_details, name="generation_details"),
    path("generations/<int:generation_id>/audio", views.generation_details, name="generation_audio"),
]