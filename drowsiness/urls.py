from django.urls import path
from .views import monitor, object_and_gender_detection_view,welcome

app_name = 'drowsiness'

urlpatterns = [
    path('monita/', monitor, name='monita'),
    path('', object_and_gender_detection_view, name='object_and_gender_detection'),
]