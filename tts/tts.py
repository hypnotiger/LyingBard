import numpy as np
import torch
from .model import Talkotron, REDUCTION_FACTOR
from pydub import AudioSegment
import re
from lyingbard.misc_utils import sliding_window
from lyingbard.models import Speaker, Generation
from pathlib import Path
# try:
#     from uwsgi import mem as uwsgi_mem
#     uwsgi_found = True
# except:
#     uwsgi_found = False

from .emojisound import emoji_sounds, SOUNDS_DIR
import emoji

torch.set_num_threads(1)

from . import fake_librosa
vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

model_path = Path(__file__).parent / "talkotron_model.pt"
model = Talkotron()
model.load_state_dict(torch.load(model_path, map_location="cpu"))

import g2p_en
phonemizer = g2p_en.G2p()

EMOJI_PLACEHOLDER = " E. .E "
EMOJI_PHONEHOLDER = ("IY1", " ", ". .", " ", "IY1")
def prep_emoji(text: str):
    matches = []
    new_text = text
    for character in emoji_sounds.keys():
        found = re.finditer(character, text, flags=re.UNICODE)
        matches.extend(found)
        new_text = new_text.replace(character, EMOJI_PLACEHOLDER)
    matches.sort(key=lambda x: x.start())
    return new_text, matches

def insert_emoji_sounds(phonemes: list[str], matches: list[re.Match], attention: np.ndarray, audio: AudioSegment) -> AudioSegment:
    insert_inds = []
    for i, phoneme in enumerate(phonemes):
        if phoneme == EMOJI_PLACEHOLDER:
            insert_ind = np.argmax(attention[i])
            insert_ind *= REDUCTION_FACTOR
            insert_ind *= 256 # stft hop size
            insert_ind += 512 # half stft window size
            insert_ind /= audio.frame_rate
            insert_ind *= 1000
            insert_inds.append((insert_ind, emoji_sounds[matches.pop(0).group()]))
    new_audio = audio
    for insert_ind, emoji_sound in reversed(insert_inds):
        new_audio = new_audio[:insert_ind] + AudioSegment.from_file(emoji_sound).set_frame_rate(22050) + new_audio[insert_ind:]
    return new_audio

def replace_emoji_placeholder(phonemes: list[str]) -> list[str]:
    for i, seq in reversed(list(enumerate(sliding_window(phonemes, len(EMOJI_PHONEHOLDER))))):
        if seq == EMOJI_PHONEHOLDER:
            phonemes = phonemes[:i] + [EMOJI_PLACEHOLDER] + phonemes[i+len(EMOJI_PHONEHOLDER):]
    return phonemes

def text_to_speech(text: str, speaker: Speaker) -> AudioSegment:
    text = emoji.emojize(text, language="alias")
    text, matches = prep_emoji(text)
    text = phonemizer(text)
    text = replace_emoji_placeholder(text)
    text = ["SOS"] + text + ["EOS"]
    with torch.no_grad():
        spectrogram, attention = model(text, torch.Tensor(speaker.speaker_embedding))
        attention = attention.cpu().numpy()[0]
        
    audio = vocoder.inverse(spectrogram*1.2)
    audio = audio.cpu().numpy()[0]

    audio *= np.iinfo(np.int16).max
    audio = audio.astype(np.int16)
    audio = AudioSegment(
        audio.tobytes(), 
        frame_rate=22050,
        sample_width=audio.dtype.itemsize,
        channels=1
    )
    audio += 6
    audio = insert_emoji_sounds(text, matches, attention, audio)
    return audio

CHUNK = 1024