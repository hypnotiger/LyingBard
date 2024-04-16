from django.db import models
from random import randint
from math import ceil
from .misc_utils import int_to_base64, base64_to_int, BASE64_CHARS
from django.core.exceptions import ValidationError
import json
from django.dispatch import receiver
from django.contrib.auth import get_user_model
from django.utils.html import escape
from storages.backends.gcloud import GoogleCloudStorage
from functools import partial
from django.apps import apps

"""
remember the three-step guide to making model changes:
    Change your models (in models.py).
    Run python manage.py makemigrations to create migrations for those changes
    Run python manage.py migrate to apply those changes to the database.
"""

class IDMixin():
    id_length = 4 # number of base64 digits

    def lazy_unique_random_id(lazy_class_ref):
        return apps.get_model(lazy_class_ref).unique_random_id()

    @classmethod
    def unique_random_id(cls) -> int:
        occupied = True
        while occupied:
            id = randint(0, 64**cls.id_length-1) # random 4 digit base64 int
            occupied = cls.objects.filter(id=id).exists()
        return id

    @property
    def url_id(self) -> str:
        return int_to_base64(self.id, length=self.id_length)
    
    @staticmethod
    def url_id_to_id(id: str) -> int:
        return base64_to_int(id)

    @classmethod
    def url_id_validator(cls, id: str):
        if cls.id_length != len(id):
            raise ValidationError("Incorrect Length")
        for char in id:
            if char not in BASE64_CHARS:
                raise ValidationError("Invalid Digit")

class Speaker(IDMixin, models.Model):
    id = models.PositiveIntegerField(primary_key=True, editable=False, default=partial(IDMixin.lazy_unique_random_id, "lyingbard.Speaker"))
    name = models.CharField(max_length=255)
    file = models.FileField(storage=GoogleCloudStorage(
        bucket_name="lyingbard-speakers"
    ))
    owner = models.ForeignKey(get_user_model(), default=None, on_delete=models.CASCADE, null=True)

    readonly_fields = ["id"]

    def __str__(self):
        username = self.owner.get_username() if self.owner != None else "Anonymous"
        return escape(f'{username}\'s {self.name}')
    
    @property
    def speaker_embedding(self) -> dict | list:
        with self.file.open() as f:
            speaker_embedding = json.load(f)
        return speaker_embedding
    
    class Meta:
        permissions = [
            ("lyingbard_speakers_create", "Can create LyingBard speakers")
        ]

class TTSModel(IDMixin, models.Model):
    id = models.PositiveIntegerField(primary_key=True, editable=False, default=partial(IDMixin.lazy_unique_random_id, "lyingbard.TTSModel"))
    name = models.CharField(max_length=255)
    file = models.FileField(storage=GoogleCloudStorage(
        bucket_name="lyingbard-models"
    ))
    speaker = models.ForeignKey(Speaker, on_delete=models.CASCADE, blank=True)
    owner = models.ForeignKey(get_user_model(), default=None, on_delete=models.CASCADE, null=True)

    readonly_fields = ["id"]

    def __str__(self):
        username = self.owner.get_username() if self.owner != None else "Anonymous"
        return escape(f'{username}\'s {self.name}')

class Generation(IDMixin, models.Model):
    id = models.PositiveIntegerField(primary_key=True, editable=False, default=partial(IDMixin.lazy_unique_random_id, "lyingbard.Generation"))
    file = models.FileField(storage=GoogleCloudStorage(
        bucket_name="lyingbard-generations"
    ))
    text = models.TextField(default="")
    gen_date = models.DateTimeField("date generated", auto_now_add=True)
    speaker = models.ForeignKey(Speaker, on_delete=models.CASCADE)
    owner = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, null=True)

    readonly_fields = ["id", "gen_date"]

    def __str__(self):
        username = self.owner.get_username() if self.owner != None else "Anonymous"
        return escape(f'{username} made {self.speaker.name} say "{self.text[:47]}{"..." if len(self.text) > 50 else ""}"')

@receiver(models.signals.post_delete, sender=TTSModel)
@receiver(models.signals.post_delete, sender=Generation)
@receiver(models.signals.post_delete, sender=Speaker)
def delete_file(sender, instance, *args, **kwargs):
    instance.file.delete(save=False)

