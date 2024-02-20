import torch
from torch import nn
import numpy as np
from scipy.stats import betabinom
from tqdm import tqdm
from math import *
from torch.signal.windows import gaussian

from string import punctuation, whitespace
CHAR_VOCAB = [
	'SOS', 'EOS', 'PAD', *list(punctuation), "...", *list(whitespace), 'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0',
	'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2', 'AO', 'AO0', 'AO1', 'AO2', 'AW',
	'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH',
	'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY', 'EY0', 'EY1', 'EY2',
	'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1', 'IY2', 'JH',
	'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0', 'OY1',
	'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
	'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]

N_MELS = 80
RNN_SIZE = 256
REDUCTION_FACTOR = 5
NUM_TOKENS = 10

TRAIN = False
if TRAIN:
	DEVICE = "cuda"
else:
	DEVICE = "cpu"

MILESTONES = []
GAMMA = 0.5
CLIP_GRAD_NORM = 0.05

def zoneout(prev, current, p=0.1):
	mask = torch.empty_like(prev).bernoulli_(p)
	return mask * prev + (1 - mask) * current

# https://en.wikipedia.org/wiki/Highway_network
# https://github.com/c0nn3r/pytorch_highway_networks/blob/master/layers/highway.py
# https://towardsdatascience.com/review-highway-networks-gating-function-to-highway-image-classification-5a33833797b5
class HighwayFullyConnectedLayer(nn.Module):
	def __init__(self, in_features: int, out_features: int, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		self.normal_layer = nn.Linear(in_features, out_features)
		self.transform_gate = nn.Sequential(
			nn.Linear(in_features, out_features),
			nn.Sigmoid()
		)
		nn.init.zeros_(self.normal_layer.bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.lerp(nn.functional.relu(self.normal_layer(x)), x, self.transform_gate(x))

class Concat(nn.Module):
	def __init__(self, *submodules: nn.Module, **kwargs) -> None:
		super().__init__(**kwargs)
		self.submodules = submodules

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.concat([submodule(x) for submodule in self.submodules])

from typing import Tuple
class CBHG(nn.Module):
	def __init__(self, K: int, projections_size: Tuple[int, int], in_channels=RNN_SIZE//2, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		self.conv1d_bank = nn.ModuleList(
			[nn.Sequential(
				nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=k, padding=k//2),
				# nn.Conv1d(in_channels=RNN_SIZE//2, out_channels=RNN_SIZE//2, kernel_size=k if k<8 else k//2+k%2, dilation=1 if k<8 else 2, padding=k//2),
				nn.ReLU(), nn.BatchNorm1d(in_channels)) for k in range(1, K+1)], # Alternating sequence of Conv1D and ReLU of length K (Conv1D bank)
			)
		self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
		self.conv1d_projections = nn.Sequential(
			nn.Conv1d(in_channels=in_channels*K, out_channels=projections_size[0], kernel_size=3, padding=1),
			nn.ReLU(),
			nn.BatchNorm1d(projections_size[0]),
			nn.Conv1d(in_channels=projections_size[0], out_channels=projections_size[1], kernel_size=3, padding=1),
			nn.BatchNorm1d(projections_size[1]),
			)
		self.highway_net = nn.Sequential(
			*[HighwayFullyConnectedLayer(in_features=projections_size[1], out_features=projections_size[1]) for _ in range(8)], # Sequence of Highway Layers length 8 (4 layers of FC-128-ReLU)
		)
		self.bidirectional_gru = nn.GRU(input_size=projections_size[1], hidden_size=projections_size[1], num_layers=1, bidirectional=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x.transpose(1, 2)
		length = x.size(-1)
		y = [conv(x)[..., :length] for conv in self.conv1d_bank]
		y = torch.cat(y, dim=1)
		y = self.max_pool(y)
		y = self.conv1d_projections(y[..., :length])
		y = x + y # Residual connection
		y = y.transpose(1, 2)
		y = self.highway_net(y)
		y, _ = self.bidirectional_gru(y)
		return y

# https://github.com/bshall/Tacotron/blob/6fee34a7c3a9d4ceb9215ed3063771a9287010e1/tacotron/model.py#L143
class DynamicConvultionAttention(nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		attn_rnn_size = 256  # should match model.decoder.attn_rnn_size
		hidden_size = 128
		static_channels = 8
		static_kernel_size = 21
		self.dynamic_channels = 8
		self.dynamic_kernel_size = 21
		self.prior_length = 11
		alpha = 0.1
		beta = 0.9

		P = betabinom.pmf(np.arange(self.prior_length), self.prior_length-1, alpha, beta)
		self.register_buffer("P", torch.FloatTensor(P).flip(0))
		self.W = nn.Linear(attn_rnn_size, hidden_size)
		self.V = nn.Linear(hidden_size, self.dynamic_channels * self.dynamic_kernel_size, bias=False)
		self.F = nn.Conv1d(
			in_channels=1,
			out_channels=static_channels,
			kernel_size=static_kernel_size,
			padding=(static_kernel_size-1) // 2,
			bias=False
		)
		self.U = nn.Linear(static_channels, hidden_size, bias=False)
		self.T = nn.Linear(self.dynamic_channels, hidden_size)
		self.v = nn.Linear(hidden_size, 1, bias=False)

	def forward(self, s, alpha: torch.Tensor):
		p = nn.functional.conv1d(
			nn.functional.pad(alpha.unsqueeze(1), (self.prior_length - 1, 0)), self.P.view(1, 1, -1)
		)
		p = torch.log(p.clamp_min_(1e-6)).squeeze(1)

		G = self.V(torch.tanh(self.W(s)))
		g = nn.functional.conv1d(
			alpha.unsqueeze(0),
			G.view(-1, 1, self.dynamic_kernel_size),
			padding=(self.dynamic_kernel_size - 1) // 2,
			groups=s.size(0),
		)
		g = g.view(s.size(0), self.dynamic_channels, -1).transpose(1, 2)

		f = self.F(alpha.unsqueeze(1)).transpose(1, 2)

		e = self.v(torch.tanh(self.U(f) + self.T(g))).squeeze(-1) + p

		return nn.functional.softmax(e, dim=-1)

# https://github.com/bshall/Tacotron/blob/main/tacotron/model.py#L341
class DecoderCell(nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		self.pre_net = nn.Sequential(
			nn.Linear(in_features=N_MELS*REDUCTION_FACTOR, out_features=RNN_SIZE), nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(in_features=RNN_SIZE, out_features=RNN_SIZE//2), nn.ReLU(),
			nn.Dropout(0.5)
		)
		self.dca = DynamicConvultionAttention()
		self.attention_rnn = nn.GRUCell(input_size=RNN_SIZE+RNN_SIZE//2+2*RNN_SIZE, hidden_size=RNN_SIZE)
		self.linear = nn.Linear(4*RNN_SIZE, RNN_SIZE)
		self.decoder_rnn1 = nn.GRUCell(input_size=RNN_SIZE, hidden_size=RNN_SIZE)
		self.decoder_rnn2 = nn.GRUCell(input_size=RNN_SIZE, hidden_size=RNN_SIZE)
		self.project = nn.Linear(RNN_SIZE, N_MELS * REDUCTION_FACTOR)



	def forward(self,
				encoder_outputs,
				prev_frame,
				prev_attention,
				decoder_cell_state,
				prev_attention_hidden_state,
				prev_decoder_rnn1_hidden_state,
				prev_decoder_rnn2_hidden_state
				):
		batch_size = prev_frame.size()[0]
		output = self.pre_net(prev_frame)
		attention_hidden_state = self.attention_rnn(
			torch.concat((decoder_cell_state, output), dim=-1),
			prev_attention_hidden_state
			)
		if self.training:
			attention_hidden_state = zoneout(prev_attention_hidden_state, attention_hidden_state, p=0.1)
		
		if not self.training:
			prev_attention[..., 1:] += prev_attention[..., :-1] * 0.1
			prev_attention /= 1.1

		attention = self.dca(attention_hidden_state, prev_attention)
		
		if not self.training:
			attention += 0.1*(attention - torch.conv1d(attention, gaussian(5, device=attention.device).unsqueeze(0).unsqueeze(0), padding=2))

		decoder_cell_state = torch.matmul(attention.unsqueeze(1), encoder_outputs).squeeze(1)

		x = self.linear(torch.concat((decoder_cell_state, attention_hidden_state), dim=1))

		decoder_rnn1_hidden_state = self.decoder_rnn1(x, prev_decoder_rnn1_hidden_state)
		if self.training:
			decoder_rnn1_hidden_state = zoneout(prev_decoder_rnn1_hidden_state, decoder_rnn1_hidden_state, p=0.1)
		x = x + decoder_rnn1_hidden_state

		decoder_rnn2_hidden_state = self.decoder_rnn2(x, prev_decoder_rnn2_hidden_state)
		if self.training:
			decoder_rnn1_hidden_state = zoneout(prev_decoder_rnn2_hidden_state, decoder_rnn2_hidden_state, p=0.1)
		x = x + decoder_rnn2_hidden_state

		output = self.project(x).view(batch_size, N_MELS, REDUCTION_FACTOR)

		return output, attention, decoder_cell_state, attention_hidden_state, decoder_rnn1_hidden_state, decoder_rnn2_hidden_state

#https://github.com/jinhan/tacotron2-gst/blob/master/modules.py
class StyleTokenModule(nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		num_channels_list = [1, RNN_SIZE//8, RNN_SIZE//8, RNN_SIZE//4, RNN_SIZE//4, RNN_SIZE//2, RNN_SIZE//2]
		self.reference_encoder_conv2d = nn.Sequential(
			*[nn.Sequential(
				nn.Conv2d(in_channels=num_channels_list[i], out_channels=num_channels_list[i+1], kernel_size=3, stride=2, padding=1),
				nn.BatchNorm2d(num_features=num_channels_list[i+1]),
				nn.ReLU()
			) for i in range(len(num_channels_list)-1)]
		)
		self.reference_encoder_gru = nn.GRU(input_size=RNN_SIZE, hidden_size=RNN_SIZE, batch_first=True)

		num_heads = 8
		style_embed_dim = RNN_SIZE*2

		self.style_embedding = nn.Parameter(torch.FloatTensor(NUM_TOKENS, style_embed_dim // num_heads))
		torch.nn.init.normal_(self.style_embedding, mean=0, std=0.5)

		d_q = RNN_SIZE
		d_k = style_embed_dim // num_heads
		self.attention = torch.nn.MultiheadAttention(embed_dim=d_q, kdim=d_k, vdim=style_embed_dim, num_heads=num_heads, batch_first=True)
		self.keys_to_value = nn.Linear(style_embed_dim // num_heads, style_embed_dim)

	def forward(self, input):
		# reference encoder
		N = input.size(0)
		out = input.reshape(N, 1, -1, 80)  # [N, 1, Ty, n_mels]

		out = self.reference_encoder_conv2d(out)

		out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
		T = out.size(1)
		N = out.size(0)
		out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

		_, out = self.reference_encoder_gru(out)  # out --- [1, N, E//2]

		out =  out.squeeze(0)

		# style token layer

		N = out.size(0)
		query = out.unsqueeze(1)  # [N, 1, E//2]
		keys = nn.functional.tanh(self.style_embedding).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]

		value = self.keys_to_value(keys)
		style_embed, combination_weights = self.attention(query, keys, value, need_weights=True)

		return style_embed, combination_weights
	
class TextPredictedGST(nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.gru = nn.GRU(input_size=RNN_SIZE, hidden_size=RNN_SIZE//4, num_layers=1, batch_first=True)
		self.linear = nn.Linear(RNN_SIZE//4, NUM_TOKENS)
	
	def forward(self, text_encoding):
		output, _ = self.gru(text_encoding)
		output = output[:, -1].unsqueeze(1)
		output = self.linear(output)
		return output

class ResidualPostNet(nn.Module):
	def __init__(self, in_size, mid_size, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.down = nn.Linear(in_size, mid_size)
		self.up = nn.Linear(mid_size, in_size)
	
	def forward(self, input):
		output = self.down(input)
		output = nn.functional.relu(output)
		output = self.up(output)
		return input + output
	
class ResidualLayer(nn.Module):
	def __init__(self, size, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.layer = nn.GRU(size, size, bidirectional=True, batch_first=True)
		self.project = nn.Linear(size*2, size)
	
	def forward(self, input):
		output = self.layer(input)[0]
		output = nn.functional.leaky_relu(output)
		output = self.project(output)
		return input + output

# https://arxiv.org/pdf/1703.10135.pdf
class Talkotron(nn.Module):
	device = "cpu"
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		# Table 1 in paper describes specific architecture implemented

		self.character_embedding = nn.Embedding(len(CHAR_VOCAB), RNN_SIZE)
		self.encoder_pre_net = nn.Sequential(
			nn.Linear(in_features=RNN_SIZE, out_features=RNN_SIZE), nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(in_features=RNN_SIZE, out_features=RNN_SIZE//2), nn.ReLU(),
			nn.Dropout(0.5)
		)
		self.encoder_cbhg = CBHG(K=16, projections_size=(RNN_SIZE//2, RNN_SIZE//2))

		self.decoder_cell = DecoderCell()
		self.gst = StyleTokenModule()
		self.tpgst = TextPredictedGST()
		self.post_net = CBHG(K=8, projections_size=[RNN_SIZE, N_MELS], in_channels=N_MELS)
		self.last_linear = nn.Linear(N_MELS * 2, N_MELS)
	
	@staticmethod
	def phone_to_tensor(phonemes: list) -> torch.Tensor:
		phonemes = [CHAR_VOCAB.index(char) if char in CHAR_VOCAB else CHAR_VOCAB.index(" ") for char in phonemes]
		phonemes = torch.tensor(phonemes, dtype=torch.long)
		return phonemes
	
	def forward(self, inputs: torch.Tensor | list, speaker_embeddings: torch.Tensor, spectrograms: torch.Tensor = None, use_tpgst:bool=True):
		train = not spectrograms==None

		if isinstance(inputs, list):
			inputs = self.phone_to_tensor(inputs)

		inputs, speaker_embeddings = inputs.to(self.device), speaker_embeddings.to(self.device)
		if train:
			spectrograms = spectrograms.to(self.device)

		if len(inputs.size()) <2:
			inputs = inputs.unsqueeze(0)
			speaker_embeddings = speaker_embeddings.unsqueeze(0)
		
		if train:
			batch_size, _, spectrogram_length = spectrograms.size()
		else:
			batch_size, _ = inputs.size()
		
		max_length = spectrogram_length if train else inputs.shape[-1] * 20
		
		inputs = self.character_embedding(inputs)
		inputs = self.encoder_pre_net(inputs)
		inputs = self.encoder_cbhg(inputs)
		encoder_outputs = inputs
		del inputs
	
		tpgst_weights = self.tpgst(encoder_outputs)
		keys = nn.functional.tanh(self.gst.style_embedding).unsqueeze(0).expand(1, -1, -1)
		tpgsts = nn.functional.linear(self.gst.keys_to_value(keys), self.gst.attention.v_proj_weight)
		tpgsts = torch.bmm(tpgst_weights, tpgsts.repeat(batch_size, 1, 1))
		
		encoder_outputs = torch.concat((encoder_outputs, speaker_embeddings.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)), dim=-1)

		if train:
			gsts = tpgsts
		else:
			if use_tpgst:
				gsts = tpgsts
			else:
				gsts, gst_weights = self.gst(spectrograms)

		encoder_outputs = torch.concat((encoder_outputs, gsts.repeat(1, encoder_outputs.size(1), 1)), dim=-1)
		del gsts

		attention = nn.functional.one_hot(
			torch.zeros(batch_size, dtype=torch.long, device=self.device), encoder_outputs.size(1)
			).float()

		init_frame = -4 * torch.ones((batch_size, N_MELS*REDUCTION_FACTOR), device=self.device)
		decoder_cell_state = torch.zeros((batch_size, 3*RNN_SIZE), device=self.device)
		attention_hidden_state, decoder_rnn1_hidden_state, decoder_rnn2_hidden_state = [torch.zeros((batch_size, RNN_SIZE), device=self.device) for _ in range(3)]
		outputs, attentions = [], []

		prev_frame = init_frame
		for prev_frame_index in range(0, max_length, REDUCTION_FACTOR):
			output, attention, decoder_cell_state, attention_hidden_state, decoder_rnn1_hidden_state, decoder_rnn2_hidden_state = self.decoder_cell(
				encoder_outputs,
				prev_frame,
				attention,
				decoder_cell_state,
				attention_hidden_state,
				decoder_rnn1_hidden_state,
				decoder_rnn2_hidden_state
				)
			if not train and torch.all(attention[:, -1] > 0.4):
				break
			outputs.append(output)
			attentions.append(attention)
			if train:
				prev_frame = spectrograms[..., prev_frame_index: prev_frame_index+REDUCTION_FACTOR]
			else:
				prev_frame = outputs[-1]
			prev_frame = prev_frame.reshape(batch_size, N_MELS*REDUCTION_FACTOR)

		outputs = torch.concat(outputs, dim=-1)
		post_outputs = outputs.transpose(1,2)
		post_outputs = self.post_net(post_outputs)
		post_outputs = self.last_linear(post_outputs)
		post_outputs = post_outputs.transpose(1,2)
		attentions = torch.stack(attentions, dim=2)

		if train:
			return outputs, post_outputs, attentions, gst_weights if not use_tpgst else None, tpgst_weights
		else:
			return post_outputs, attentions
		
	def to(self, device, *args, **kwargs):
		self = super().to(device=device, *args, **kwargs)
		self.device = device
		return self

from pathlib import Path
from typing import Callable
from functools import partial
class TrainStruct:
	model: Talkotron
	optimizer: torch.optim.Optimizer
	scaler: torch.cuda.amp.grad_scaler.GradScaler
	step: int = 0
	_save_fn: Callable
	
	def __init__(self, *args, **kwargs):
		self.model = Talkotron()
		self.model = self.model.train()
		self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
		super().__init__(*args, **kwargs)
	
	def initialize(self, save_fn:Callable=partial(torch.save, f="model.pt"), optimizer:torch.optim.Optimizer = torch.optim.Adamax, *args, **kwargs):
		if TRAIN:
			self.n_steps = 30000
			self.lr = 1e-3
			self.checkpoint_interval = 2000
		else:
			self.n_steps = 1000
			self.lr = 5e-5
			self.checkpoint_interval = 1000
		self._save_fn = save_fn
		if not TRAIN:
			self.optimizer = optimizer(self.get_parameters(), lr=self.lr)
			self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.7)
		else:
			self.optimizer = optimizer(self.get_parameters(), lr=self.lr)
			self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, MILESTONES, GAMMA)

		self.n_steps = self.n_steps if "n_steps" not in kwargs else kwargs.pop("n_steps")
		self.lr = self.lr if "lr" not in kwargs else kwargs.pop("lr")
		self.checkpoint_interval = self.checkpoint_interval if "checkpoint_interval" not in kwargs else kwargs.pop("checkpoint_interval")
		self.lr_scheduler = None

	def save(self): self._save_fn(self.model.state_dict())

	def get_parameters(self):
		model = self.model
		modules = [
			{
				"params": [
					model.character_embedding,
					model.encoder_pre_net,
					model.encoder_cbhg,
					*[
						model.decoder_cell.pre_net,
						model.decoder_cell.dca,
						model.decoder_cell.attention_rnn,
						model.decoder_cell.linear,
						model.decoder_cell.decoder_rnn1,
						model.decoder_cell.decoder_rnn2,
						model.decoder_cell.project,
					]
				],
				"lr": self.lr
			},
			{
				"params": [
					model.post_net,
					model.last_linear,
				],
				"lr": self.lr
			}
		]
		for i in range(len(modules)):
			parameters = sum([list(m.parameters()) for m in modules[i]["params"]], [])
			modules[i]["params"] = parameters
		return modules
	
	def _load_stripped(self, state_dict: dict, *args, **kwargs):
		self.model.load_state_dict(state_dict)
		self.model = self.model.to(DEVICE)
		self.initialize(*args, **kwargs)
	
	def _load_unstripped(self, state_dict: dict):
		self.step = state_dict["step"]
		self._load_stripped(state_dict["tacotron"])
		optimizer_state_dict = self.optimizer.state_dict()
		optimizer_state_dict["state"] = state_dict["optimizer"]["state"]
		milestone_lr = self.get_milestone_lr(self.step, self.lr, MILESTONES, GAMMA)
		for g in optimizer_state_dict["param_groups"]:
			g['lr'] = milestone_lr
		self.optimizer.load_state_dict(optimizer_state_dict)

	@staticmethod
	def load(load_path: Path, *args, **kwargs):
		self = TrainStruct()
		state_dict: dict = torch.load(load_path)
		if "tacotron" in state_dict.keys():
			self._load_unstripped(state_dict, *args, **kwargs)
		else:
			self._load_stripped(state_dict, *args, **kwargs)
		return self
	
	@staticmethod
	def get_milestone_lr(step: int, learning_rate: float, milestones: list[int], gamma: float):
		milestone_lr = learning_rate
		for milestone in milestones:
			if milestone <= step:
				milestone_lr *= gamma
			else:
				break
		return milestone_lr

def train_model(dataset, train_struct: TrainStruct):
	if TRAIN:
		n_epochs = ceil(train_struct.n_steps / len(dataset))
	else:
		n_epochs = ceil(train_struct.n_steps / len(dataset.iterable))

	for epoch in tqdm(range(0, n_epochs + 1)):
		average_loss = 0

		for i, (spectrograms, texts, speaker_embeddings) in enumerate(tqdm(dataset, leave=False)):
			texts, spectrograms, speaker_embeddings = texts.to(train_struct.model.device), spectrograms.to(train_struct.model.device), speaker_embeddings.to(train_struct.model.device)

			train_struct.optimizer.zero_grad()

			outputs, post_outputs, _, _, _ = train_struct.model(texts, speaker_embeddings, spectrograms=spectrograms, use_tpgst=not TRAIN)
			loss = torch.nn.functional.l1_loss(outputs, spectrograms)
			post_loss = torch.nn.functional.l1_loss(post_outputs, spectrograms)

			loss += post_loss
			if train_struct.model.device == "cuda" and TRAIN:
				train_struct.scaler.scale(loss).backward()
				train_struct.scaler.unscale_(train_struct.optimizer)
				torch.nn.utils.clip_grad.clip_grad_norm_(train_struct.model.parameters(), CLIP_GRAD_NORM)
				train_struct.scaler.step(train_struct.optimizer)
				train_struct.scaler.update()
			else:
				loss.backward()
				train_struct.optimizer.step()

			train_struct.step += 1
			if train_struct.lr_scheduler:
				train_struct.lr_scheduler.step()

			average_loss += (loss.item() - average_loss) / (i+1)

			if train_struct.step % train_struct.checkpoint_interval == 0:
				train_struct.save()
			if train_struct.step >= train_struct.n_steps:
				train_struct.save()
				return

def collate_fn(batch):
	spectrograms, transcripts, speaker_embeddings = zip(*batch)
	spectrograms, transcripts, speaker_embeddings = list(spectrograms), list(transcripts), list(speaker_embeddings)
	spectrograms = [spectrogram.T for spectrogram in spectrograms]

	longest_ind, longest_len = max(enumerate([spectrogram.shape[0] for spectrogram in spectrograms]),key=lambda x: x[1])
	remainder = longest_len % REDUCTION_FACTOR
	if remainder != 0:
		spectrograms[longest_ind] = torch.nn.functional.pad(spectrograms[longest_ind], (0, 0, 0, REDUCTION_FACTOR - remainder), value=0)

	spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True, padding_value=0)
	transcripts = torch.nn.utils.rnn.pad_sequence(transcripts, batch_first=True, padding_value=CHAR_VOCAB.index("PAD"))

	speaker_embeddings = torch.stack(speaker_embeddings, dim=0)

	return spectrograms.transpose(1,2), transcripts, speaker_embeddings