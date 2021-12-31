import os
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from pathlib import Path 

folder_base_url = Path(__file__).resolve().parent.parent

folders = {
	'contour': {
		'1': 'test\\output\\contours\\1\\'
	},
	'res'	 : {
		'1': 'test\\output\\res\\1\\'
	},
	'input'  : 'static\\test\\input\\'
}

def filewatcher():
	input_folder = os.path.join(folder_base_url, folders['input'])
	inputfiles = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))])
	return inputfiles


def index(request):
	data=filewatcher()
	context = {
		'inputs' : data
	}
	template = loader.get_template('polls/index.html')
	return HttpResponse(template.render(context, request))


