from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import shutil
from pathlib import Path
from .wizard import Wizard
from .models import TTS

class TTSName(Frame):
    name: StringVar

    def __init__(self, parent):
        super().__init__(parent)

        header = Label(self, text="Name your TTS.", bd=2)
        header.pack(side=TOP, fill=X, pady=4)

        self.name = StringVar(self)
        name_entry = Entry(self, textvariable=self.name)
        name_entry.pack(side=TOP)

class TTSAudio(Frame):
    filenames = list[str]

    def __init__(self, parent):
        super().__init__(parent)

        header = Label(self, text="Load your audio.", bd=2)
        header.pack(side=TOP, fill=X, pady=4)

        self.filenames = []
        def load_audio_files():
            self.filenames = filedialog.askopenfilenames(title="Load Audio")
        button = Button(self, text="Load Audio", command=load_audio_files)
        button.pack(side=TOP)

class TTSTranscribe(Frame):
    filename: str
    transcribe: Text

    def __init__(self, parent, filename):
        super().__init__(parent)

        self.filename = filename

        header = Label(self, text=f"{Path(self.filename).name}", bd=2)
        header.pack(side=TOP, fill=X, pady=4)

        self.transcribe = Text(self, wrap=WORD)
        self.transcribe.pack(side=TOP)

def create_tts(name: str, filenames: list[str], transcriptions: list[str]):
    tts: TTS = TTS.new(name, filenames, transcriptions)
    tts.tune()
    tts.save()

def tts_wizard(parent):
    def finish():
        name = name_step.name.get()
        filenames = audio_step.filenames
        def finish_transcribe():
            transcriptions = [transcribe.transcribe.get("1.0", END) for transcribe in transcribes]
            create_tts(name, filenames, transcriptions)
        transcribe_wizard = Wizard(parent, finish_transcribe)
        transcribes = [TTSTranscribe(transcribe_wizard, filename) for filename in audio_step.filenames]
        for transcribe in transcribes:
            transcribe_wizard.add_step(transcribe)
        transcribe_wizard.start()
    wizard = Wizard(parent, finish)
    name_step = TTSName(wizard)
    audio_step = TTSAudio(wizard)
    wizard.add_step(name_step)
    wizard.add_step(audio_step)
    wizard.start()


def retrain_wizard(parent):
    window = Toplevel(parent)
    window.geometry("240x240")
    
    tts_select = ttk.Combobox(window, values=TTS.list_all(), state="readonly")
    tts_select.place(relx=0.5, rely=0.5, anchor=CENTER)

    def retrain():
        tts = TTS(tts_select.get())
        tts.tune()
        tts.save()
        window.destroy()
    
    submit = Button(window, command=retrain, text="Retrain")
    submit.pack(side=BOTTOM, pady=1)