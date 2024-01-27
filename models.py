from django.db import models
from random import randint
from math import ceil
from .misc_utils import int_to_base64, base64_to_int, BASE64_CHARS
from django.core.exceptions import ValidationError
import json
from django.dispatch import receiver
from django.contrib.auth import get_user_model
from django.utils.html import escape

"""
remember the three-step guide to making model changes:
    Change your models (in models.py).
    Run python manage.py makemigrations to create migrations for those changes
    Run python manage.py migrate to apply those changes to the database.
"""

class IDMixin():
    id_length = 4 # number of base64 digits

    def save(self):
        if not self.id:
            self.id = self.unique_random_id()
        super().save()

    @classmethod
    def unique_random_id(model) -> int:
        occupied = True
        while occupied:
            id = randint(0, 64**model.id_length-1) # random 4 digit base64 int
            occupied = model.objects.filter(id=id).exists()
        return id

    @property
    def url_id(self) -> str:
        return int_to_base64(self.id, length=self.id_length)
    
    @staticmethod
    def url_id_to_id(id: str) -> int:
        return base64_to_int(id)

    @classmethod
    def url_id_validator(model, id: str):
        if model.id_length != len(id):
            raise ValidationError("Incorrect Length")
        for char in id:
            if char not in BASE64_CHARS:
                raise ValidationError("Invalid Digit")

class Speaker(IDMixin, models.Model):
    id = models.PositiveIntegerField(primary_key=True, editable=False)
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to="lyingbard_speakers")
    owner = models.ForeignKey(get_user_model(), default=None, on_delete=models.CASCADE, null=True)

    def __str__(self):
        username = self.owner.get_username() if self.owner != None else "Anonymous"
        return escape(f'{username}\'s {self.name}')
    
    @property
    def speaker_embedding(self) -> dict | list:
        with self.file.open() as f:
            speaker_embedding = json.load(f)
        return speaker_embedding

class Generation(IDMixin, models.Model):
    id = models.PositiveIntegerField(primary_key=True, editable=False)
    file = models.FileField(upload_to="lyingbard_generations")
    text = models.TextField(default="")
    gen_date = models.DateTimeField("date generated", auto_now_add=True)
    speaker = models.ForeignKey(Speaker, on_delete=models.CASCADE)
    owner = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, null=True)

    def __str__(self):
        username = self.owner.get_username() if self.owner != None else "Anonymous"
        return escape(f'{username} made {self.speaker.name} say "{self.text[:47]}{"..." if len(self.text) > 50 else ""}"')

@receiver(models.signals.post_delete, sender=Generation)
@receiver(models.signals.post_delete, sender=Speaker)
def delete_file(sender, instance, *args, **kwargs):
    instance.file.delete(save=False)

