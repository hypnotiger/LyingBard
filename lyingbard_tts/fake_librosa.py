# Stupid MelGAN imports librosa to do absolutely nothing. So I have to make a fake dummy one to stop import errors.

from types import ModuleType
from sys import modules
import numpy as np
from importlib.machinery import ModuleSpec

fake_librosa = ModuleType("librosa")
fake_librosa.__spec__ = ModuleSpec("librosa", None)

def dummy_mel(*args, **kwargs):
    return np.zeros(0)

fake_librosa_filters = ModuleType("librosa.filters")
fake_librosa.filters = fake_librosa_filters
fake_librosa_filters.__spec__ = ModuleSpec("librosa.filters", None)
fake_librosa_filters.mel = dummy_mel

def pad_center(*args, **kwargs):
    return np.zeros(0)

fake_librosa_util = ModuleType("librosa.util")
fake_librosa.util = fake_librosa_util
fake_librosa_util.__spec__ = ModuleSpec("librosa.util", None)
fake_librosa_util.pad_center = pad_center

modules["librosa"] = fake_librosa
modules["librosa.filters"] = fake_librosa_filters
modules["librosa.util"] = fake_librosa_util