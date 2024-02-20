from django.contrib import admin

from .models import Speaker, Generation, TTSModel
admin.site.register(Speaker)
admin.site.register(Generation)
admin.site.register(TTSModel)