from django.shortcuts import render
from django.http import HttpResponse, Http404
from django.template import loader
from .models import Speaker, Generation

def index(request):
    return HttpResponse(render(request, "lyingbard/index.html"))

def generate(request):
    context = {
        "speakers": Speaker.objects.all(),
    }
    return HttpResponse(render(request, "lyingbard/generate.html"))

def speaker_details(request, speaker_id):
    return HttpResponse(f"Speaker: {speaker_id}")

def generation_details(request, generation_id):
    return HttpResponse(f"Lyingbard generation: {generation_id}")

def generation_audio(request, generation_id):
    return Http404()