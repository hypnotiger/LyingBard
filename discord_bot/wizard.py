from lyingbard.wizard import Wizard
from tkinter import *
from tkinter import ttk
import platformdirs
from pathlib import Path

DISCORD_DIR = Path(platformdirs.user_data_dir("LyingBard", "FUBAI"))/"Bot"/"Discord"

class DiscordToken(Frame):
    token: StringVar

    def __init__(self, parent):
        super().__init__(parent)

        header = Label(self, text=f"Enter your Discord bot's Token.", bd=2)
        header.pack(side=TOP, fill=X, pady=4)

        self.token = StringVar(self)
        token_entry = Entry(self, show="*", textvariable=self.token)
        token_entry.pack(side=TOP)

def setup_wizard(parent):
    def finish():
        (DISCORD_DIR).mkdir(exist_ok=True, parents=True)
        with open(DISCORD_DIR/"Token.txt", "w") as file:
            file.write(auth_token.token.get())

    wizard = Wizard(parent, finish)
    auth_token = DiscordToken(wizard)
    wizard.add_step(auth_token)
    wizard.start()