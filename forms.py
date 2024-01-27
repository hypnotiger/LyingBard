from django import forms
from .models import Speaker, Generation
from .misc_utils import base64_to_int
from django.core.exceptions import ValidationError

class B64IDField(forms.CharField):
    def __init__(self, length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = length

    def clean(self, value):
        if self.length != len(value):
            raise ValidationError("Wrong Length B64ID")
        try:
            return base64_to_int(value)
        except:
            raise ValidationError("Not Valid Base64")

class GenerateForm(forms.Form):
    speaker = B64IDField(Speaker.id_length)
    text = forms.fields.CharField()

class GenerationDeleteForm(forms.Form):
    id = B64IDField(Generation.id_length)