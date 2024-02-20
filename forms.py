from typing import Any
from django import forms
from django.forms.boundfield import BoundField
from django.forms.forms import BaseForm
from .models import TTSModel, Generation
from .misc_utils import base64_to_int
from django.core.exceptions import ValidationError
from django.core.files import File
from django.template.defaultfilters import filesizeformat
from pydub import AudioSegment
from io import BytesIO

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
    model = B64IDField(TTSModel.id_length)
    text = forms.fields.CharField()

class GenerationDeleteForm(forms.Form):
    id = B64IDField(Generation.id_length)

class ContentTypeRestrictedFileField(forms.FileField):
    """
    Same as FileField, but you can specify:
        * content_types - list containing allowed content_types. Example: ['application/pdf', 'image/jpeg']
        * max_upload_size - a number indicating the maximum file size allowed for upload.
            2.5MB - 2621440
            5MB - 5242880
            10MB - 10485760
            20MB - 20971520
            50MB - 5242880
            100MB - 104857600
            250MB - 214958080
            500MB - 429916160
    """
    def __init__(self, *args, **kwargs):
        self.content_types = kwargs.pop("content_types", [])
        self.max_upload_size = kwargs.pop("max_upload_size", 0)

        super(ContentTypeRestrictedFileField, self).__init__(*args, **kwargs)

    def clean(self, *args, **kwargs):
        data = super(ContentTypeRestrictedFileField, self).clean(*args, **kwargs)

        file: File = data.file
        try:
            content_type = file.content_type
            if content_type in self.content_types:
                if file._size > self.max_upload_size:
                    raise forms.ValidationError(('Please keep filesize under %s. Current filesize %s') % (filesizeformat(self.max_upload_size), filesizeformat(file.size)))
            else:
                raise forms.ValidationError(('Filetype not supported.'))
        except AttributeError:
            pass

        return data

class AudioFileField(ContentTypeRestrictedFileField):
    def __init__(self, *args, **kwargs):
        kwargs["content_types"] = ["audio/wav", "audio/mpeg", "audio/ogg", "audio/flac"]
        super().__init__(*args, **kwargs)
    
    def clean(self, *args, **kwargs):
        data = super(AudioFileField, self).clean(*args, **kwargs)
        file: BytesIO = data.file
        try:
            return AudioSegment.from_file(file)
        except:
            raise ValidationError("Could not read file as audio.")

class MultipleAudioFileField(AudioFileField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def clean(self, data, *args, **kwargs):
        single_file_clean = super().clean
        print(data)
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, *args, **kwargs) for d in data]
        else:
            result = single_file_clean(data, *args, **kwargs)
        return result
    
    def get_bound_field(self, form: BaseForm, field_name: str) -> BoundField:
        if form.files[field_name] != form.files.getlist(field_name)[0]:
            form_files = form.files.copy()
            form_files[field_name] = form.files.getlist(field_name)
            form.files = form_files
        bound_field = super().get_bound_field(form, field_name)
        return bound_field
    
class ListField(forms.Field):
    def __init__(self, subfield, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subfield: forms.Field = subfield
    
    def get_bound_field(self, form: BaseForm, field_name: str) -> BoundField:
        if form.data[field_name] != form.data.getlist(field_name)[0]:
            form_data = form.data.copy()
            form_data[field_name] = form.data.getlist(field_name)
            form.data = form_data
        bound_field = super().get_bound_field(form, field_name)
        return bound_field

    def clean(self, data):
        print(data)
        if data == None:
            raise ValidationError("Fields are empty")
        try:
            data = [self.subfield.clean(d) for d in data]
        except:
            raise ValidationError("Field not iterable")
        return data

class SpeakerCreateForm(forms.Form):
    audio = MultipleAudioFileField(allow_empty_file=False, max_upload_size=1e6)
    name = forms.CharField(max_length=255)
    transcript = ListField(subfield=forms.CharField())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)