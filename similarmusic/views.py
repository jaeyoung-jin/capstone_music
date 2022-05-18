import csv
import numpy as np
import librosa
import pandas as pd
import sklearn

from django.shortcuts import render
from django.http import HttpResponse
from . import models

from .models import Dataset, Music #DB 모델 사용 가능하도록 import

import compare_music
import compare_newmusic

# Create your views here.

def data_view(request):
    data = Dataset.objects.all()
    music = Music.objects.all()
    return render(request, 'index.html',{"data":data})


def play_music(request):
    music = Music.objects.all()
    return render(request, 'show-files.html',{"music":music})

def uploadFile(request):
    if request.method == "POST":
        # Fetching the form data
        fileTitle = request.POST.get("fileTitle")
        uploadedFile = request.FILES["uploadedFile"]

        # Saving the information in the database
        document = models.Document(
            title = fileTitle,
            uploadedFile = uploadedFile
            
        )
        document.save()

    documents = models.Document.objects.all()

    return render(request, "similarmusic/upload-file.html", context = {
        "files": documents
    })


def play_music2(request):
    musics = compare_newmusic.compare_music()
    return render(request, 'show-files.html',{"musics":musics})

def test(request):
    name = compare_music.say_hello("재영")
    return render(request,'test.html',{"name":name} )

