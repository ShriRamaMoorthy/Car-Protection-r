from django.http import StreamingHttpResponse
from .camera import VideoCamera, gen
from django.shortcuts import render,redirect,reverse


def monitor(request):
    print("hi")
    return StreamingHttpResponse(gen(VideoCamera()), content_type='multipart/x-mixed-replace; boundary=frame')

def findface_view(request):
    return render(request, 'authorization/auth.html')

def summa_redirect(request):
    return redirect(reverse('drowsiness:object_and_gender_detection'))