import torch
from torch import nn
import numpy as np
from scipy.stats import betabinom

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
DEVICE = "cpu"

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

        attention = self.dca(attention_hidden_state, prev_attention)

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

    def train_forward(self, inputs: torch.Tensor, spectrograms: torch.Tensor, speaker_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.character_embedding(inputs)
        inputs = self.encoder_pre_net(inputs)
        inputs = self.encoder_cbhg(inputs)
        encoder_outputs = inputs
        del inputs

        tpgst_weights = self.tpgst(encoder_outputs)

        encoder_outputs = torch.concat((encoder_outputs, speaker_embeddings.repeat(1, encoder_outputs.size(1), 1)), dim=-1)
        del speaker_embeddings

        gst_outputs, gst_weights = self.gst(spectrograms)
        encoder_outputs = torch.concat((encoder_outputs, gst_outputs.repeat(1, encoder_outputs.size(1), 1)), dim=-1)
        del gst_outputs

        batch_size, _, spectrogram_length = spectrograms.size()

        attention = nn.functional.one_hot(
            torch.zeros(batch_size, dtype=torch.long, device=DEVICE), encoder_outputs.size(1)
            ).float()

        init_frame = -torch.ones((batch_size, N_MELS*REDUCTION_FACTOR), device=DEVICE)
        decoder_cell_state = torch.zeros((batch_size, 3*RNN_SIZE), device=DEVICE)
        attention_hidden_state, decoder_rnn1_hidden_state, decoder_rnn2_hidden_state = [torch.zeros((batch_size, RNN_SIZE), device=DEVICE) for _ in range(3)]
        outputs, attentions = [], []

        for prev_frame_index in range(-1, spectrogram_length-1, REDUCTION_FACTOR):
            prev_frame = spectrograms[..., prev_frame_index-REDUCTION_FACTOR+1: prev_frame_index+1].reshape(batch_size, N_MELS*REDUCTION_FACTOR) if prev_frame_index != -1 else init_frame
            output, attention, decoder_cell_state, attention_hidden_state, decoder_rnn1_hidden_state, decoder_rnn2_hidden_state = self.decoder_cell(
                encoder_outputs,
                prev_frame,
                attention,
                decoder_cell_state,
                attention_hidden_state,
                decoder_rnn1_hidden_state,
                decoder_rnn2_hidden_state
                )
            outputs.append(output)
            attentions.append(attention)

        outputs = torch.concat(outputs, dim=-1)
        outputs = outputs.transpose(1,2)
        outputs = self.post_net(outputs)
        outputs = self.last_linear(outputs)
        outputs = outputs.transpose(1,2)
        attentions = torch.stack(attentions, dim=2)

        return outputs, attentions, gst_weights, tpgst_weights

    def forward(self, text: torch.Tensor, speaker_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(text, list):
            text = [CHAR_VOCAB.index(char) if char in CHAR_VOCAB else CHAR_VOCAB.index(" ") for char in text]
            text = torch.tensor(text, dtype=torch.long, device=DEVICE)
        inputs = text
        max_length = inputs.shape[-1] * 20

        if len(inputs.size()) <2:
            inputs = inputs.unsqueeze(0)
            speaker_embeddings = speaker_embeddings.unsqueeze(0)
        batch_size = inputs.size(0)

        inputs = self.character_embedding(inputs)
        inputs = self.encoder_pre_net(inputs)
        inputs = self.encoder_cbhg(inputs)
        encoder_outputs = inputs
        del inputs
        
        tpgst_weights = self.tpgst(encoder_outputs)
        # print(tpgst_weights)
        keys = nn.functional.tanh(self.gst.style_embedding).unsqueeze(0).expand(1, -1, -1)
        gsts = nn.functional.linear(self.gst.keys_to_value(keys), self.gst.attention.v_proj_weight)
        gsts = torch.bmm(tpgst_weights, gsts)

        encoder_outputs = torch.concat((encoder_outputs, speaker_embeddings.repeat(1, encoder_outputs.size(1), 1)), dim=-1)
        
        encoder_outputs = torch.concat((encoder_outputs, gsts.repeat(1, encoder_outputs.size(1), 1)), dim=-1)

        attention = nn.functional.one_hot(
            torch.zeros(batch_size, dtype=torch.long, device=DEVICE), encoder_outputs.size(1)
            ).float()

        init_frame = -4 * torch.ones((batch_size, N_MELS*REDUCTION_FACTOR), device=DEVICE)
        decoder_cell_state = torch.zeros((batch_size, 3*RNN_SIZE), device=DEVICE)
        attention_hidden_state, decoder_rnn1_hidden_state, decoder_rnn2_hidden_state = [torch.zeros((batch_size, RNN_SIZE), device=DEVICE) for _ in range(3)]
        outputs, attentions = [], []

        stop_threshold = 0.5
        for prev_frame_index in range(-1, max_length-1, REDUCTION_FACTOR):
            prev_frame = outputs[-1].reshape(batch_size, N_MELS*REDUCTION_FACTOR) if prev_frame_index > 0 else init_frame
            output, attention, decoder_cell_state, attention_hidden_state, decoder_rnn1_hidden_state, decoder_rnn2_hidden_state = self.decoder_cell(
                encoder_outputs,
                prev_frame,
                attention,
                decoder_cell_state,
                attention_hidden_state,
                decoder_rnn1_hidden_state,
                decoder_rnn2_hidden_state
                )
            if torch.all(attention[:, -1] > stop_threshold):
                break
            outputs.append(output)
            attentions.append(attention)

        outputs = torch.concat(outputs, dim=-1)
        outputs = outputs.transpose(1,2)
        outputs = self.post_net(outputs)
        outputs = self.last_linear(outputs)
        outputs = outputs.transpose(1,2)
        attentions = torch.stack(attentions, dim=2)

        return outputs, attentions