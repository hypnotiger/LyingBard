from torch import Tensor
from .networks.speaker_embedder import SpeakerEmbedderGRU
from .networks.convrnn_embedder import ConvRNNEmbedder, ConvRNNConfig
import torch
import torch.nn as nn

class ConvGRUEmbedder(nn.Module):

    def __init__(self, device='cuda') -> None:
        super().__init__()
        self.device = device
        self.model = ConvRNNEmbedder(ConvRNNConfig(), mode='eval').to(device)
        self.model_params = self.model.cfg
        self.model.eval()
    
    def forward(self, x: Tensor) -> Tensor:
        """ Takes in set of 16kHz waveforms (bs, n_samples) and returns (bs, 256) speaker embeddings """
        x = x.clamp(-1.0 , 1.0)
        x = x.to(self.device)
        lengths = torch.ones((x.shape[0],), dtype=torch.long, device=x.device)*x.shape[-1]
        x = x.unsqueeze(1) # account for (utters per speaker) dimension 
        return self.model(x, lengths).squeeze(1)

    def print_hparams(self):
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(self.model_params))

    @staticmethod
    def strip_checkpoint(ckpt_path: str, out_path: str) -> None:
        """
        Strips a checkpoint of unnecessary data for inference.
        i.e. it strips out the optimizer state dict and other training
        data that is not needed for inference.
        Path of full checkpoint is `ckpt_path`, and the target output path is `out_path`.
        NOTE: you will need to import TrainConfig and DistributedConfig from train.py 
        in your top-level script before calling this due to pickle being a pickle.
        """        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        del ckpt['loss_fn_state_dict']
        del ckpt['optim_state_dict']
        del ckpt['scheduler_state_dict']
        del ckpt['scaler_state_dict']
        torch.save(ckpt, out_path)
        print(f"Stripped {ckpt_path} and saved to {out_path}")


def convgru_embedder(pretrained=True, progress=True, device='cuda'):
    r""" 
    GRU embedding model trained on the VCTK, Librispeech, and VoxCeleb 1 & 2 datasets.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        device (str): device to load model onto ('cuda' or 'cpu' are common choices)
    """
    model = ConvGRUEmbedder(device=device)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url("https://github.com/RF5/simple-speaker-embedding/releases/download/v1.0/convgru_ckpt_00700000_strip.pt", 
                                                progress=progress, map_location=device)
        model.model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

    return model

def convgru_embedder_sc09(pretrained=True, progress=True, device='cuda'):
    r""" 
    GRU embedding model trained on the Google Speech Commands digits dataset.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        device (str): device to load model onto ('cuda' or 'cpu' are common choices)
    """
    model = ConvGRUEmbedder(device=device)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url("https://github.com/RF5/simple-speaker-embedding/releases/download/0.1/convgru_sc09_ckpt_00077500-strip.pt", 
                                                progress=progress, map_location=device)
        model.model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

    return model
