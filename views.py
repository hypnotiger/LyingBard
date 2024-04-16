from django import shortcuts
from django.http import HttpResponse, HttpResponseBadRequest, FileResponse, JsonResponse, HttpResponseRedirect
from django.template import loader
from django.shortcuts import get_object_or_404
from .models import Speaker, Generation, TTSModel
from .forms import GenerateForm, GenerationDeleteForm, SpeakerCreateForm, SpeakerDeleteForm
from .cloud_tts import text_to_speech_and_save, finetune_and_save
from io import BytesIO, TextIOWrapper
from django.urls import reverse, reverse_lazy
from django.contrib.auth import decorators
from .discord import bot
import json

def render(*args, **kwargs):
    subpage = {
        "subpage_name": "LyingBard",
        "subpage_url": "lyingbard:index"
    }
    kwargs["context"] = {
        **(kwargs["context"] if "context" in kwargs else {}),
        **subpage
    }
    return shortcuts.render(*args, **kwargs)

def index(request):
    return render(request, "lyingbard/index.html")

def generate(request):
    match request.method:
        case "GET":
            return render(request, "lyingbard/generate/index.html")

def speaker_details(request, speaker_id):
    speaker = Speaker.objects.filter(id=speaker_id)
    if not speaker.exists(): return HttpResponseBadRequest("Speaker does not exist.")
    speaker = speaker.first()
    if speaker.file.storage.exists(speaker.file.name):
        return render(request, "lyingbard/speakers/details.html", context={
            "speaker":speaker,
            "subpage_name": "LyingBard",
            "subpage_url": "lyingbard:index"
        })
    else:
        return render(request, "lyingbard/speakers/submitted.html", context={
            "speaker_id":speaker_id,
            "subpage_name": "LyingBard",
            "subpage_url": "lyingbard:index"
        })

@decorators.login_required
def speakers(request):
    return render(request, "lyingbard/speakers/index.html")

@decorators.login_required
def speakers_widget(request):
    speakers=Speaker.objects.filter(owner=request.user)
    return render(request, "lyingbard/speakers/widget.html", context={
        "speakers":speakers
    })

@decorators.login_required
def speaker_delete(request):
    if request.method != "POST":
        return
    form = SpeakerDeleteForm(request.POST)
    if not form.is_valid():
        return
    speaker = Speaker.objects.filter(id=form.cleaned_data["id"])
    if not speaker.exists():
        return
    speaker = speaker.first()
    if request.user != speaker.owner:
        return
    speaker.delete()
    return HttpResponseRedirect(reverse("lyingbard:speakers_widget"))

@decorators.permission_required("lyingbard.lyingbard_speakers_create", reverse_lazy("donate:subscription"))
@decorators.login_required
def speaker_create(request):
    return render(request, "lyingbard/speakers/create.html")

@decorators.permission_required("lyingbard.lyingbard_speakers_create", reverse_lazy("donate:subscription"))
@decorators.login_required
def speaker_create_widget(request):
    match request.method:
        case "GET":
            return render(request, "lyingbard/speakers/create_widget.html")
        case "POST":
            form = SpeakerCreateForm(request.POST, request.FILES)
            if not form.is_valid():
                return render(request, "lyingbard/speakers/create_widget.html", context={"form":form})
            
            speaker, model = finetune_and_save(form.cleaned_data["name"], form.files["audio"], form.cleaned_data["transcript"], request.user)

            return HttpResponseRedirect(reverse("lyingbard:speaker_details", args=(speaker.id,)))

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
            user = request.user if request.user.is_authenticated else None

            try:
                generation = text_to_speech_and_save(form.cleaned_data["text"], model, speaker, user)
                return JsonResponse({
                "audioURL": reverse("lyingbard:generation_audio", args=(generation.id,))
                })
            except:
                response = JsonResponse({
                "message": "The LyingBard generation server is down right now. Please come back in a bit."
                })
                response.status_code = 503
                return response

def generation_details(request, generation_id):
    return HttpResponse(f"Lyingbard generation: {generation_id}")

def generation_audio(request, generation_id):
    generation = get_object_or_404(Generation, id=generation_id)
    return FileResponse(generation.file.open(), content_type="audio/flac")

@decorators.login_required
def generations_widget(request):
    return render(request, "lyingbard/generations/widget.html", context={
        "generations": request.user.generation_set.all().order_by("-gen_date")
    })

@decorators.login_required
def generations(request):
    return

@decorators.login_required
def generation_delete(request):
    if request.method != "POST":
        return
    form = GenerationDeleteForm(request.POST)
    if not form.is_valid():
        return
    generation = Generation.objects.get(id=form.cleaned_data["id"])
    if not generation.exists():
        return
    generation = generation.first()
    if request.user != generation.owner:
        return
    generation.delete()
    return JsonResponse({})

def discord_bot_invite(request):
    return HttpResponseRedirect("https://discord.com/api/oauth2/authorize?client_id=1201237999959167068&permissions=3147776&scope=bot+applications.commands")

def discord_bot(request):
    return render(request, "lyingbard/discord_bot.html")