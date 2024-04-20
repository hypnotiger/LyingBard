from tkinter import *
from typing import Callable

# https://stackoverflow.com/a/41338656/23530805
class Wizard(Toplevel):
    def __init__(self, parent, finish_fn: Callable, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.current_step = None
        self.steps = []

        self.button_frame = Frame(self, bd=1, relief="raised")
        self.content_frame = Frame(self)

        self.finish_fn = finish_fn

        self.back_button = Button(self.button_frame, text="<< Back", command=self.back)
        self.next_button = Button(self.button_frame, text="Next >>", command=self.next)
        self.finish_button = Button(self.button_frame, text="Finish", command=self.finish)

        self.button_frame.pack(side=BOTTOM, fill=X)
        self.content_frame.pack(side=TOP, fill=BOTH, expand=True)

        self.geometry("240x240")

    def add_step(self, step: Frame): self.steps.append(step)

    def start(self): self.show_step(0)
    def back(self): self.show_step(self.current_step-1)
    def next(self): self.show_step(self.current_step+1)
    def finish(self):
        self.finish_fn()
        self.destroy()

    def show_step(self, step):

        if self.current_step is not None:
            # remove current step
            current_step = self.steps[self.current_step]
            current_step.pack_forget()

        self.current_step = step

        new_step = self.steps[step]
        new_step.pack(fill="both", expand=True)

        if step == 0:
            self.back_button.pack_forget()
        else:
            self.back_button.pack(side=LEFT)
        
        if step == len(self.steps)-1:
            self.next_button.pack_forget()
            self.finish_button.pack(side=RIGHT)
        else:
            self.finish_button.pack_forget()
            self.next_button.pack(side=RIGHT)