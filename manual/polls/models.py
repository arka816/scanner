from django.db import models
from django.contrib.postgres.fields import ArrayField

class Votes(models.Model):
	input_image_url = models.CharField(max_length=200, default="", blank=False)
	width = models.IntegerField(default=0)
	height = models.IntegerField(default=0)

	edge_detection_choices = [
		('1', '1')
	]
	edge_detection_votes = ArrayField(models.IntegerField(default=0), size = len(edge_detection_choices))
	edge_output_image_url = ArrayField(models.CharField(max_length=200, default="", blank=False), size=len(edge_detection_choices), blank=True, null=True)

	contour_detection_choices = [
		('1', '1')
	]
	contour_detection_votes = ArrayField(models.IntegerField(default=0), size = len(contour_detection_choices))	
	contour_output_image_url = ArrayField(models.CharField(max_length=200, default="", blank=False), size=len(contour_detection_choices), blank=True, null=True)
