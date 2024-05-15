import numpy as np
import random
import json
from finetuning import SupplementalDataset
from pathlib import Path

def main():
    dataset = SupplementalDataset.new(Path(__file__).parent/"train-clean-100-npy"/"spectrograms.npy", Path(__file__).parent/"train-clean-100-npy"/"info.json", 1)
    
    indexes = random.sample(range(dataset._true_size), 500)

    truncated_dataset = [dataset.get_true_item(i) for i in indexes]

    item_list = [dataset.info["item_list"][i] for i in indexes]
    unique_speakers = set([item["speaker"] for item in item_list])

    spectrograms = np.concatenate([i[0].T for i in truncated_dataset])
    np.save("500/spectrograms.npy", spectrograms)

    current_ind = 0
    for item in item_list:
        prev_start, prev_end = item["spectrogram_inds"]
        length = prev_end - prev_start
        item["spectrogram_inds"] = [current_ind, current_ind+length]
        current_ind += length

    info = {
		"speaker_embeddings": {
			speaker: dataset.info["speaker_embeddings"][speaker]
            for speaker in unique_speakers
		},
		"item_list": item_list
    }
    with open("500/info.json", "w") as file:
        json.dump(info, file)

if __name__ == "__main__":
    main()