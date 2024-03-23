import cv2
import torch
from PIL import Image
import torchvision.transforms
import numpy as np
from django.shortcuts import render
from django.http import StreamingHttpResponse
from .camera import VideoCamera, gen


def monitor(request):
    return StreamingHttpResponse(gen(VideoCamera()), content_type='multipart/x-mixed-replace; boundary=frame')
    

def object_and_gender_detection_view(request):
    return render(request, 'drowsiness/index.html')

def welcome(request):
    return render(request, 'drowsiness/welcome.html')