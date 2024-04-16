TTS_ADDRESS = "http://127.0.0.1:5000"

def text_to_speech_and_save(text:str, model:TTSModel, speaker:Speaker, user):
    generation = Generation(text=text, speaker=speaker, owner=user)
    text_to_speech(text, model, speaker, generation.url_id)
    generation.file.name = f"{generation.url_id}.flac"
    generation.save()
    return generation

def finetune(audios: list[File], transcripts: list[str], model_id: str, speaker_id: str):
    for audio in audios:
        audio.file.seek(0)
    try:
        response = requests.post(TTS_ADDRESS+"/finetune", {
                "transcripts": transcripts,
                "model_id": model_id,
                "speaker_id": speaker_id
            },
            files=[("audio", audio.file) for audio in audios],
            timeout=1e-6
        )
    except:
        pass

def finetune_and_save(name: str, audios: list[File], transcripts: list[str], user) -> Generation:
    speaker = Speaker(name=name, owner=user)
    speaker.file.name = f"{speaker.url_id}.json"
    speaker.save()

    model = TTSModel(name=name, speaker=speaker, owner=user)
    model.file.name = f"{model.url_id}.pt"
    model.save()

    finetune(audios, transcripts, model.url_id, speaker.url_id)

    return speaker, model