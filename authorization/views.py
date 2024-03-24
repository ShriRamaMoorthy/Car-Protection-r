from django.http import StreamingHttpResponse
from .camera import VideoCamera, gen
from django.shortcuts import render,redirect,reverse
from django.http import JsonResponse


def monitor(request):
    print("hi")
    return StreamingHttpResponse(gen(VideoCamera()), content_type='multipart/x-mixed-replace; boundary=frame')

def findface_view(request):
    return render(request, 'authorization/auth.html')

def summa_redirect(request):
    return redirect(reverse('drowsiness:object_and_gender_detection'))

def process_frame(request):
    if request.method == 'POST' and request.is_ajax():
        # Get the image data from the POST request
        image_data = request.POST.get('image_data')

        # Process the image data here (e.g., perform face recognition)

        # Return a JSON response indicating success
        return JsonResponse({'message': 'Frame processed successfully'}, status=200)
    else:
        # Return a JSON response indicating error
        return JsonResponse({'error': 'Invalid request'}, status=400)