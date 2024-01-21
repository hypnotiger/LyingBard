from django.db import models
from random import randint
from math import ceil
from .misc_utils import int_to_base64, base64_to_int

"""
remember the three-step guide to making model changes:
    Change your models (in models.py).
    Run python manage.py makemigrations to create migrations for those changes
    Run python manage.py migrate to apply those changes to the database.
"""

class IDMixin():
    def save(self):
        if not self.id:
            self.id = self.unique_random_id()
        super().save()

    @classmethod
    def unique_random_id(model) -> int:
        occupied = True
        while occupied:
            id = randint(0, 64**4-1) # random 4 digit base64 int
            occupied = model.objects.filter(id=id).exists()
        return id
    
    LENGTH = 4 # in bytes (b256)
    # length conversion from b256 (8 bits per digit) to b64 (6 bits per digit)
    UNPADDED_LENGTH = ceil(8 * LENGTH / 6)
    PADDED_LENGTH = UNPADDED_LENGTH + (4 - (UNPADDED_LENGTH % 4))

    def get_url_id(self) -> str:
        return int_to_base64(self.id)
    
    @staticmethod
    def url_id_to_id(id: str) -> int:
        return base64_to_int(id)

class Speaker(IDMixin, models.Model):
    id = models.PositiveIntegerField(primary_key=True, editable=False)
    name = models.CharField(max_length=255)
    file = models.FileField()

    def __str__(self):
        return self.name

class Generation(IDMixin, models.Model):
    id = models.PositiveIntegerField(primary_key=True, editable=False)
    file = models.FileField()
    text = models.TextField(default="")
    gen_date = models.DateTimeField("date generated")
    speaker = models.ForeignKey(Speaker, on_delete=models.CASCADE)
    # owner = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.speaker} says "{self.text[:50]}{"..." if len(self.text) > 50 else ""}"'

