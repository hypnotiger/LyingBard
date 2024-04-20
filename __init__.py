from tkinter import *
from tkinter import ttk
from tts_wizard import tts_wizard, retrain_wizard
from models import TTS, DATA_DIR
from pydub.playback import _play_with_pyaudio
import threading
from showinfm import show_in_file_manager

def main(root: Tk):
    menubar = Menu(root)
    root["menu"] = menubar
    menu_train = Menu(menubar)
    menu_train.add_command(label="Train", command=lambda: tts_wizard(root))
    menu_train.add_command(label="Retrain", command=lambda: retrain_wizard(root))
    menu_train.add_command(label="Show Folder", command=lambda: show_in_file_manager(str(DATA_DIR)))
    menubar.add_cascade(menu=menu_train, label='Model')

    tts: TTS = None
    def speak():
        nonlocal tts
        selected_speaker = speaker_combo.get()
        if tts == None and selected_speaker == "":
            return
        if tts == None or tts.name != selected_speaker:
            tts = TTS(selected_speaker)
            tts.load()
        text_to_speak = text.get("1.0", END)
        audio = tts.speak(text_to_speak)
        threading.Thread(target=_play_with_pyaudio, args=(audio,)).start()

    frame = ttk.Frame(root)
    content = ttk.Frame(frame, padding=(10,10))
    content['borderwidth'] = 2
    content['relief'] = 'sunken'
    text = Text(content)
    text.insert("1.0", "This is LyingBard, A T T S based off of LyreBird.")
    sidebar = ttk.Frame(frame, width=100)
    button = Button(sidebar, text="Speak", command=speak)
    speaker_combo = ttk.Combobox(sidebar, state="readonly", values=TTS.list())

    frame.grid(column=0, row=0, sticky=NSEW)
    content.grid(column=0, row=0, sticky=NSEW)
    text.grid(column=0, row=0, sticky=NSEW)
    sidebar.grid(column=1, row=0, sticky=NS)
    sidebar.grid_propagate(False)
    button.grid(column=0, row=0, sticky=EW)
    speaker_combo.grid(column=0, row=1, sticky=EW)

    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    content.rowconfigure(0, weight=1)
    content.columnconfigure(0, weight=1)
    sidebar.columnconfigure(0, weight=1)

    root.geometry("720x270")
    root.title("LyingBard")
    root.iconbitmap("lyingbard/favicon.ico")

    root.mainloop()

if __name__ == "__main__":
    root = Tk()
    root.option_add('*tearOff', FALSE)
    def show_error(self, exception, traceback):
        from tkinter.messagebox import ERROR, Message
        from traceback import format_exception
        error = format_exception(type(exception).__name__, value=exception, tb=traceback)
        error = "".join(error)
        if "slave" in error.lower():
            error = "This error contains headass language because programmers were/are pieces of shit.\n\n" + error
        message = Message(root, icon=ERROR, title="Python Error", message=error)
        message.show()
    root.report_callback_exception = show_error

    main(root)