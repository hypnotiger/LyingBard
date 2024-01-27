from django.shortcuts import render
from django.http import HttpResponse, Http404, HttpResponseBadRequest, FileResponse, JsonResponse
from django.template import loader
from django.shortcuts import get_object_or_404
from django.core.files.base import File
from .models import Speaker, Generation
from .forms import GenerateForm, GenerationDeleteForm
from .tts.tts import text_to_speech
from io import BytesIO
from django.urls import reverse, reverse_lazy
from django.core.files.storage import default_storage
from django.contrib.auth import decorators

def index(request):
    return HttpResponse(render(request, "lyingbard/index.html"))

def generate(request):
    match request.method:
        case "GET":
            return HttpResponse(render(request, "lyingbard/generate/index.html"))

def speaker_details(request, speaker_id):
    return HttpResponse(f"Speaker: {speaker_id}")

def generation_details(request, generation_id):
    return HttpResponse(f"Lyingbard generation: {generation_id}")

def generation_audio(request, generation_id):
    generation = get_object_or_404(Generation, id=generation_id)
    return FileResponse(generation.file.open(), content_type="audio/flac")

@decorators.login_required
def dashboard(request):
    return render(request, "lyingbard/dashboard.html")

def generate_widget(request):
    match request.method:
        case "GET":
            return render(request, "lyingbard/generate/widget.html", context = {
                "speakers": Speaker.objects.all(),
                })
        case "POST":
            form = GenerateForm(request.POST)
            if not form.is_valid(): return HttpResponseBadRequest()
            speaker = Speaker.objects.filter(id=form.cleaned_data["speaker"])
            if not speaker.exists(): return HttpResponseBadRequest("Speaker Does Not Exist")
            speaker = speaker.first()
            audio = text_to_speech(form.cleaned_data["text"], speaker)
            audio_file = File(audio.export(BytesIO(), 'flac'))
            user = request.user if request.user.is_authenticated else None
            generation = Generation(text=form.cleaned_data["text"], speaker=speaker, owner=user)
            generation.save()
            generation.file.save(f"{generation.url_id}.flac", audio_file)
            return JsonResponse({
                "audioURL": reverse("lyingbard:generation_audio", args=(generation.id,))
                })

def generations_widget(request):
    return render(request, "lyingbard/generations/widget.html", context={
        "generations": request.user.generation_set.all().order_by("-gen_date")
    })

def generations(request):
    return

def generations_delete(request):
    if request.method != "POST":
        return
    form = GenerationDeleteForm(request.POST)
    if not form.is_valid():
        return
    generation = Generation.objects.get(id=form.cleaned_data["id"])
    generation.delete()
    return JsonResponse({})