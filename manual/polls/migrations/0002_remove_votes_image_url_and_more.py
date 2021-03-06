# Generated by Django 4.0 on 2021-12-13 19:33

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='votes',
            name='image_url',
        ),
        migrations.AddField(
            model_name='votes',
            name='contour_output_image_url',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.CharField(default='', max_length=200), blank=True, null=True, size=1),
        ),
        migrations.AddField(
            model_name='votes',
            name='edge_output_image_url',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.CharField(default='', max_length=200), blank=True, null=True, size=1),
        ),
        migrations.AddField(
            model_name='votes',
            name='input_image_url',
            field=models.CharField(default='', max_length=200),
        ),
    ]
