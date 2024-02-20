import numpy as np
import torch
from .model import Talkotron, REDUCTION_FACTOR, DEVICE
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, normalize
import re
from lyingbard.misc_utils import sliding_window
from lyingbard.models import Speaker, TTSModel
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import os
import random
from scipy import ndimage, fftpack
# try:
#     from uwsgi import mem as uwsgi_mem
#     uwsgi_found = True
# except:
#     uwsgi_found = False

from .emojisound import emoji_sounds, SOUNDS_DIR
import emoji

# supposed to make torch run deterministically
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
torch.use_deterministic_algorithms(True, warn_only=True)

from . import fake_librosa
vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

default_model_path = Path(__file__).parent / "talkotron_model.pt"
current_model = Talkotron().to(DEVICE)
current_model = current_model.eval()
current_ttsmodel = None
def load_model(model: TTSModel):
    global current_ttsmodel
    current_ttsmodel = model
    current_model.load_state_dict(torch.load(model.file.open(), map_location="cpu"))
    model.file.close()
    print(f"loaded model {current_ttsmodel}")

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

def grapheme_to_phoneme(text):
    text = emoji.emojize(text, language="alias")
    text, matches = prep_emoji(text)
    text = phonemizer(text)
    text = replace_emoji_placeholder(text)
    text = ["SOS"] + text + ["EOS"]
    return text, matches

def text_to_speech(text: str, model: TTSModel, speaker: Speaker) -> AudioSegment:
    if model != current_ttsmodel:
        load_model(model)
    text, matches = grapheme_to_phoneme(text)
    with torch.no_grad():
        spectrogram, attention = current_model(text, torch.Tensor(speaker.speaker_embedding))
    
    spectrogram = spectrogram.cpu().numpy()
    # https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html
    # filtering out reduction factor noise
    # spec_fft = fftpack.fft(spectrogram)
    # plt.imshow(np.abs(spec_fft)[0], norm=LogNorm(vmin=5))
    # plt.show()
    # spec_fft[..., REDUCTION_FACTOR-1:REDUCTION_FACTOR+2] = 0.01
    # spec_fft[..., -REDUCTION_FACTOR-1:-REDUCTION_FACTOR+2] = 0.01
    # plt.figure()
    # plt.imshow(np.abs(spec_fft)[0], norm=LogNorm(vmin=5))
    # plt.figure()
    # plt.plot(spec_fft[0,40])
    # plt.yscale('log')
    # plt.show()
    # plt.yscale('linear')
    # spec_fft = fftpack.ifft(spec_fft).real
    # spectrogram = spec_fft
    # plt.imshow(spectrogram[0])
    # plt.show()
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
    # audio = normalize(audio)
    # audio += 5
    audio = insert_emoji_sounds(text, matches, attention, audio)
    return audio

CHUNK = 1024