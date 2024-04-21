import torch
from pydub import AudioSegment

from . import fake_librosa
from .simple_speaker_embedding.hubconf import convgru_embedder
speaker_embedder = convgru_embedder(device='cpu')
speaker_embedder.cpu()
speaker_embedder.eval()

def embed(audio: AudioSegment):
    audio = audio.set_frame_rate(16000)
    audio_data = torch.Tensor(audio.get_array_of_samples()).unsqueeze(0)
    with torch.no_grad():
        speaker_embedding: torch.Tensor = speaker_embedder(audio_data).squeeze(0)
    return speaker_embedding