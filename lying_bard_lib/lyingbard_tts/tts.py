import numpy as np
import torch
from .model import REDUCTION_FACTOR
from pydub import AudioSegment
import re
from pathlib import Path
import os
import random
from scipy import ndimage
from .model import Talkotron
from matplotlib import pyplot as plt

#from .emojisound import emoji_sounds
#import emoji

# From itertools recipes
# Generalized pairwise
import collections
from itertools import islice
def sliding_window(iterable, n):
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = collections.deque(islice(it, n-1), maxlen=n)
    for x in it:
        window.append(x)
        yield tuple(window)

# supposed to make torch run deterministically
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
# torch.set_num_threads(1)
# os.environ["OMP_NUM_THREADS"]="1"
# os.environ["MKL_NUM_THREADS"]="1"
# torch.use_deterministic_algorithms(True, warn_only=True)

from . import fake_librosa
from .melgan_neurips.hubconf import load_melgan
vocoder = load_melgan()

DEFAULT_MODEL_PATH = Path(__file__).parent / "talkotron_model.pt"

import g2p_en
phonemizer = g2p_en.G2p()
'''
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

def grapheme_to_phoneme(text: str):
    text = text.replace('\n', ' ')
    print(text)
    text = emoji.emojize(text, language="alias")
    text, matches = prep_emoji(text)
    text = phonemizer(text)
    text = replace_emoji_placeholder(text)
    text = ["SOS"] + text + ["EOS"]
    return text, matches

def text_to_speech(text: str, model: Talkotron, speaker: torch.Tensor) -> AudioSegment:
    text, matches = grapheme_to_phoneme(text)
    with torch.no_grad():
        spectrogram, attention = model(text, speaker)
    
    spectrogram = spectrogram.cpu().numpy()
    # https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_sharpen.html
    spectrogram = spectrogram + 0.5*(spectrogram - ndimage.gaussian_filter(spectrogram, 1, axes=(1,2)))
    spectrogram = torch.from_numpy(spectrogram)
    audio = vocoder.inverse(spectrogram*1.2)
    audio = audio.cpu().numpy()[0]
    attention = attention.cpu().numpy()[0]

    audio *= np.iinfo(np.int16).max
    audio = audio.astype(np.int16)
    audio = AudioSegment(
        audio.tobytes(), 
        frame_rate=22050,
        sample_width=audio.dtype.itemsize,
        channels=1
    )
    # audio = insert_emoji_sounds(text, matches, attention, audio)
    return audio
'''
