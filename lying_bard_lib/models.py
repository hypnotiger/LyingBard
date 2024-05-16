from .lyingbard_tts.model import Talkotron
import platformdirs
import torch
from torch import Tensor
from pathlib import Path
import json
from .lyingbard_tts import tts
from .lyingbard_tts.finetuning import finetune
from pydub import AudioSegment
import shutil
from typing import Self

DATA_DIR = Path(platformdirs.user_data_dir("LyingBard", "FUBAI"))
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

class TTS():
    path: Path
    model: Talkotron
    speaker: Tensor

    def __init__(self, name: str = "") -> None:
        self.path = DATA_DIR/"TTS"/name
        self.model = Talkotron()
        self.model.to(self.model.device)
        self.speaker = torch.zeros(256)

    def load(self):
        self.model.load_state_dict(
            torch.load(self.path/"model.pt")
        )
        self.model = self.model.eval()
        with open(self.path/"speaker.json") as speaker:
            self.speaker = torch.Tensor(
                json.load(speaker)
            )
    
    def change(self, name: str):
        if self.model == None and name == "":
            return
        if self.model == None or self.name != name and TTS.usable(name):
            self.path = DATA_DIR/"TTS"/name
            self.load()

    @property
    def name(self) -> str:
        if self.path == DATA_DIR/"TTS":
            return None
        else:
            return self.path.name

    def speak(self, text: str) -> AudioSegment:
        return tts.text_to_speech(text, self.model, self.speaker)
    
    def save(self):
        torch.save(
            self.model.state_dict(),
            self.path/"model.pt"
        )
        with open(self.path/"speaker.json", "w") as file:
            json.dump(self.speaker.tolist(), file)

    def new(name: str, audios: list[str], transcripts: list[str]) -> Self:
        self = TTS(name)
        (self.path/"data").mkdir(parents=True, exist_ok=True)

        transcript_json = {}
        for audio, transcript in zip(audios, transcripts):
            audio_name = Path(audio).name
            dest = self.path/f"data/{audio_name}"
            shutil.copy2(audio, dest)
            transcript_json[audio_name] = transcript

        with open(self.path/"data/transcripts.json", "w") as file:
            json.dump(transcript_json, file)

        return self

    def tune(self):
        transcripts: dict[str: str]
        with open(self.path/"data/transcripts.json") as file:
            transcripts = json.load(file)
        
        audios: list[AudioSegment] = []
        for file in transcripts.keys():
            audios.append(AudioSegment.from_file(self.path/"data"/file))

        self.model, self.speaker = finetune(audios, transcripts.values())
    
    def list_all() -> list[str]: return [tts.name for tts in (DATA_DIR/"TTS").glob("*")]
    def list_usable() -> list[str]:
        return [Path(*tts.parts[:-1]).name for tts in (DATA_DIR/"TTS").glob("*/*.pt")]
    
    def exists(name: str) -> bool:
        return (DATA_DIR/"TTS"/name).exists()

    def usable(name: str) -> bool:
        return (DATA_DIR/"TTS"/name/"model.pt").exists()
