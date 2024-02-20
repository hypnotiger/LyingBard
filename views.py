from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest, FileResponse, JsonResponse, HttpResponseRedirect
from django.template import loader
from django.shortcuts import get_object_or_404
from django.core.files.base import File
from .models import Speaker, Generation, TTSModel
from .forms import GenerateForm, GenerationDeleteForm, SpeakerCreateForm
from .tts.tts import text_to_speech
from io import BytesIO, TextIOWrapper
from django.urls import reverse, reverse_lazy
from django.contrib.auth import decorators
from .discord import bot
import json
from .tts.finetuning import finetune

def index(request):
    return HttpResponse(render(request, "lyingbard/index.html"))

def generate(request):
    match request.method:
        case "GET":
            return HttpResponse(render(request, "lyingbard/generate/index.html"))

def speaker_details(request, speaker_id):
    return HttpResponse(f"Speaker: {speaker_id}")

@decorators.login_required
def speaker_create(request):
    match request.method:
        case "GET":
            return render(request, "lyingbard/speakers/create.html")
        case "POST":
            form = SpeakerCreateForm(request.POST, request.FILES)
            if not form.is_valid():
                return render(request, "lyingbard/speakers/create.html", context={"form":form})
            model_file, speaker_embedding = finetune(form.cleaned_data["audio"], form.cleaned_data["transcript"])
            model_file = File(model_file)

            speaker_file = TextIOWrapper(BytesIO())
            json.dump(speaker_embedding.tolist(), speaker_file)
            speaker_file = File(speaker_file)

            speaker = Speaker(name=form.cleaned_data["name"], file=speaker_file, owner=request.user)
            speaker.save()
            speaker.file.save(f"{speaker.url_id}.json", speaker_file)

            model = TTSModel(name=form.cleaned_data["name"], file=model_file, speaker=speaker, owner=request.user)
            model.save()
            model.file.save(f"{model.url_id}.pt", model_file)

            return HttpResponse(f"created speaker")

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
                "models": TTSModel.objects.all().order_by("name"),
                })
        case "POST":
            form = GenerateForm(request.POST)
            if not form.is_valid(): return HttpResponseBadRequest()
            model = TTSModel.objects.filter(id=form.cleaned_data["model"])
            if not model.exists(): return HttpResponseBadRequest("Speaker Does Not Exist")
            model = model.first()
            speaker = model.speaker
            audio = text_to_speech(form.cleaned_data["text"], model, speaker)
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

def discord_bot_invite(request):
    return HttpResponseRedirect("https://discord.com/api/oauth2/authorize?client_id=1201237999959167068&permissions=3147776&scope=bot+applications.commands")

def discord_bot(request):
    return render(request, "lyingbard/discord_bot.html")