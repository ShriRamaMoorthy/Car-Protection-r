from django.urls import path,include
from django.contrib import admin
from .camera import VideoCamera, gen
from .views import findface_view,monitor,summa_redirect


urlpatterns = [
    path('admin/', admin.site.urls),
    path('monitor/', monitor, name='monitor'),
    path('', findface_view, name='findface'),
    path('summa/', summa_redirect,name="drowsy")
]