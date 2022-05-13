import csv
from django.shortcuts import render
from django.http import HttpResponse
from . import models

from .models import Dataset, Music #DB 모델 사용 가능하도록 import

# Create your views here.

def data_view(request):
    data = Dataset.objects.all()
    return render(request, 'index.html',{"data":data})


def play_music(request):
    music = Music.objects.all()
    return render(request, 'show-files.html',{"music":music})

def uploadFile(request):
    if request.method == "POST":
        # Fetching the form data
        fileTitle = request.POST["fileTitle"]
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



