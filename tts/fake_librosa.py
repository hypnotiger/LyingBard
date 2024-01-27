# Stupid MelGAN imports librosa to do absolutely nothing. So I have to make a fake dummy one to stop import errors.

from types import ModuleType
from sys import modules
import numpy as np
from importlib.machinery import ModuleSpec
fake_librosa = ModuleType("librosa")
fake_librosa_filters = ModuleType("librosa.filters")
def dummy_mel(*args, **kwargs):
    return np.zeros(0)
fake_librosa.__spec__ = ModuleSpec("librosa", None)
fake_librosa.filters = fake_librosa_filters
fake_librosa_filters.__spec__ = ModuleSpec("librosa.filters", None)
fake_librosa_filters.mel = dummy_mel
modules["librosa"] = fake_librosa
modules["librosa.filters"] = fake_librosa_filters