from django.urls import path

from . import views

app_name = "scripts"

urlpatterns = [
    path("upload/", views.upload_file, name="upload_file"),
]
