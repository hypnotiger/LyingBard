import torch
from pathlib import Path
from model import Talkotron

directory = Path(__file__).parent
talkotron_state_dict = torch.load(directory/"model.pt", map_location="cpu")["tacotron"]
torch.save(talkotron_state_dict, directory/"talkotron_model.pt")

test_model = Talkotron()
test_model.load_state_dict(talkotron_state_dict)
test_state_dict = test_model.state_dict()
def dict_compare(a, b):
    total_bool = True
    for key in a.keys():
        if isinstance(a[key], dict):
            total_bool &= dict_compare(a[key], b[key])
        else:
            total_bool &= torch.all(a[key] == b[key])
    return total_bool
print(dict_compare(talkotron_state_dict, test_state_dict))