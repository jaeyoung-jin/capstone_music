from . import views
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

app_name = "similarmusic"

urlpatterns = [
    path("", views.uploadFile, name = "uploadFile"),
    path('data/', views.data_view, name="showing data"),
    path('music/', views.play_music, name = "playing music"),
]

if settings.DEBUG: 
    urlpatterns += static(
        settings.MEDIA_URL, 
        document_root = settings.MEDIA_ROOT
    )


#바깥에 있는 urls.py랑 뭐가다르지??