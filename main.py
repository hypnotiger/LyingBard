from tkinter import *
from lying_bard_lib import tts_wizard, wizard

root = Tk()
tts_wizard.tts_wizard(root)
root.withdraw()
root.mainloop()
