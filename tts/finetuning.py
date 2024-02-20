from dataclasses import dataclass
from pydub import AudioSegment
import torch
import torchaudio
import numpy as np
from .tts import grapheme_to_phoneme
from .model import train_model, TrainStruct, Talkotron, collate_fn, DEVICE
from pathlib import Path
from .speaker import embed
from io import BytesIO
import json
import random
from multiprocessing import cpu_count
from math import *

SAMPLE_RATE = 22050
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024
N_MELS = 80
F_MIN = 0

from pydub.silence import detect_leading_silence
trim_leading_silence = lambda audio: audio[detect_leading_silence(audio) :]
trim_trailing_silence = lambda audio: trim_leading_silence(audio.reverse()).reverse()
strip_silence = lambda audio: trim_trailing_silence(trim_leading_silence(audio))

def audio_to_np(audio: AudioSegment, sample_rate: int):
	audio = audio.set_frame_rate(SAMPLE_RATE)
	audio = trim_leading_silence(audio)
	audio: np.ndarray = np.array(audio.get_array_of_samples())
	audio_dtype = audio.dtype
	audio = audio.astype(np.float32)
	audio /= np.iinfo(audio_dtype).max
	audio /= np.max(np.abs(audio))
	return audio

MEL_BASIS = torchaudio.functional.melscale_fbanks(
	sample_rate=SAMPLE_RATE, n_freqs=N_FFT//2+1, n_mels=N_MELS, f_min=F_MIN, f_max=SAMPLE_RATE/2, norm='slaney', mel_scale='slaney'
)
HANN_WINDOW = torch.hann_window(WIN_LENGTH).float()
def audio_to_mel(audio: AudioSegment): # https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py#L26
	# Audio Preprocessing
	audio = audio_to_np(audio, SAMPLE_RATE)
	audio *= 0.95
	# Spectrogram
	p = (N_FFT - HOP_LENGTH) // 2
	audio = np.pad(audio, (p, p), mode="reflect")
	audio = torch.Tensor(audio)
	fft = torch.stft(
		audio,
		n_fft=N_FFT,
		hop_length=HOP_LENGTH,
		win_length=WIN_LENGTH,
		window=HANN_WINDOW,
		center=False,
		return_complex=True
	)
	magnitude = torch.abs(fft)
	mel_output = torch.matmul(MEL_BASIS.T, magnitude)
	log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
	return log_mel_spec

@dataclass
class FineTuningDataset():
	spectrograms: list[torch.Tensor]
	transcripts: list[torch.Tensor]
	speaker_embedding: torch.Tensor

	@staticmethod
	def new(audios: list[AudioSegment], transcripts: list[str]):
		spectrograms = [audio_to_mel(audio) for audio in audios]
		transcripts = [Talkotron.phone_to_tensor(grapheme_to_phoneme(transcript)[0]) for transcript in transcripts]
		speaker_embedding = torch.from_numpy(np.mean(np.concatenate([
			embed(audio)[None]
			for audio in audios]), axis=0))
		speaker_embedding /= torch.norm(speaker_embedding)
		return FineTuningDataset(spectrograms, transcripts, speaker_embedding)

	def __len__(self) -> int:
		return len(self.spectrograms)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		spectrogram = self.spectrograms[index]
		transcript = self.transcripts[index]
		return spectrogram, transcript, self.speaker_embedding

@dataclass
class SupplementalDataset():
	spectrograms: np.ndarray
	info: dict
	sample_size: int

	"""
	Info structure
	{
		speaker_embeddings: {
			speaker: embedding
		},
		item_list: {
			phonemes: phonemes,
			spectrogram_inds: [start, end],
			speaker: speaker,
			length: length in seconds,
		}
	}
	"""
	def __post_init__(self):
		self._true_size: int = len(self.info["item_list"])
		# self.index_mapping: dict = [random.randint(0, self._true_size-1) for _ in range(self.sample_size)]

	@staticmethod
	def new(spectrograms_path: Path, info_path: Path, sample_size: int):
		with open(info_path, "r") as file:
			info = json.load(file)
		spectrograms = np.load(spectrograms_path, mmap_mode='r')
		return SupplementalDataset(spectrograms, info, sample_size)

	def __len__(self) -> int:
		return self.sample_size

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		if index >= self.sample_size:
			raise IndexError
		# index = self.index_mapping[index]
		index = random.randint(0, self._true_size-1)
		item = self.info["item_list"][index]
		start, end = item["spectrogram_inds"]
		spectrogram = torch.from_numpy(self.spectrograms[start: end].T.copy())
		transcript = Talkotron.phone_to_tensor(item["phonemes"])
		speaker_embedding = torch.tensor(self.info["speaker_embeddings"][item["speaker"]])
		return spectrogram, transcript, speaker_embedding

class ChainDataset():
	def __init__(self, *iterables) -> None:
		self.iterables = iterables
	
	def __len__(self):
		return sum([len(iterable) for iterable in self.iterables])
	
	def get_iterable_index(self, index:int):
		if index < 0:
			index += len(self)
		for i, iterable in enumerate(self.iterables):
			if index-len(iterable) < 0:
				return index, i
			index -= len(iterable)
		raise IndexError

	def __getitem__(self, index: int):
		index, i = self.get_iterable_index(index)
		return self.iterables[i][index]

class Batcher():
	def __init__(self, iterable, batch_size, collate_fn=lambda x:x) -> None:
		self.iterable = iterable
		self.batch_size: int = batch_size
		self.collate_fn = collate_fn
	
	def __len__(self):
		return floor(len(self.iterable) / self.batch_size)
	
	def __getitem__(self, index:int):
		if index < 0:
			index += len(self)
		if index >= len(self):
			raise IndexError
		start = index*self.batch_size
		end = min((index+1)*self.batch_size, len(self.iterable))
		batch_slice = [self.iterable[i] for i in range(start,end)]
		batch = self.collate_fn(batch_slice)
		return batch
	
def finetune(audios: list[AudioSegment], transcripts: list[str]):
	ft_dataset = FineTuningDataset.new(audios, transcripts)
	sup_dataset = SupplementalDataset.new(Path(__file__).parent/"train-clean-100-npy"/"spectrograms.npy", Path(__file__).parent/"train-clean-100-npy"/"info.json", len(ft_dataset)*10)
	dataset = ChainDataset(ft_dataset, sup_dataset)
	dataset = Batcher(dataset, 8, collate_fn)
	print("dataset prepared")
	train_struct = TrainStruct.load(Path(__file__).parent/"talkotron_model.pt", save_fn=lambda _:None, n_steps=500)
	print("original model loaded")
	train_model(dataset, train_struct)
	print("model finetuned")
	model = BytesIO()
	torch.save(train_struct.model.state_dict(), model)
	return model, ft_dataset.speaker_embedding