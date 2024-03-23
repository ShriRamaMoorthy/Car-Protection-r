from django.urls import path
from . import views

urlpatterns = [
    path('webcam', views.webcam, name='webcam'),
    path('', views.start_capture, name='start_capture'),
    path('stop_camera/', views.stop_camera, name='stop_camera'),
]
