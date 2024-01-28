import torch
from pathlib import Path

directory = Path(__file__).parent
talkotron_state_dict = torch.load(directory/"model.pt", map_location="cpu")["tacotron"]
torch.save(talkotron_state_dict, directory/"talkotron_model.pt")