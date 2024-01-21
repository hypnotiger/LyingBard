# Generated by Django 5.0.1 on 2024-01-21 08:42

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Speaker',
            fields=[
                ('id', models.PositiveIntegerField(editable=False, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=255)),
                ('file', models.FileField(upload_to='')),
            ],
        ),
        migrations.CreateModel(
            name='Generation',
            fields=[
                ('id', models.PositiveIntegerField(editable=False, primary_key=True, serialize=False)),
                ('file', models.FileField(upload_to='')),
                ('text', models.TextField(default='')),
                ('gen_date', models.DateTimeField(verbose_name='date generated')),
                ('speaker', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='lyingbard.speaker')),
            ],
        ),
    ]
