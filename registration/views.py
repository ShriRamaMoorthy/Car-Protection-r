from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
from .camera import VideoCamera, gen
import cv2
import os
import subprocess
import threading
# Define a global variable to keep track of the capture status
capture_active = False

def webcam(request):
    return StreamingHttpResponse(gen(VideoCamera()), content_type='multipart/x-mixed-replace; boundary=frame')

def train_model():
    # Run the model_train.py script
    subprocess.run(["python", "registration/model_train.py"])
    print("model is being trained")
    

def start_capture(request):
    global capture_active, username
    
    if request.method == 'POST':
        username = request.POST.get('username')
        # Create directory with username if it doesn't exist
        user_dir = os.path.join('frames', username)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        capture_active = True
        capture_frames_from_camera(request)
        return render(request, 'registration/model_training.html')

    return render(request, 'registration/register.html')

def stop_camera(request):
    global capture_active
    capture_active = False
    call_model_train_script(request)

    return render(request,'registration/model_training.html')

def call_model_train_script(request):
    # Run the model_train.py script
    # subprocess.run(["python", "registration/model_train.py"])
    model_thread = threading.Thread(target=train_model)
    model_thread.start()

    return render(request,'registration/model_training.html')
    

def capture_frames_from_camera(request):
    global username
    # Initialize video capture from webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    frame_count = 0
    while capture_active:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Check if frame is captured successfully
        if not ret:
            break
        
        # Increment frame count
        frame_count += 1
        
        # Save frame every 100 frames
        if frame_count % 10 == 0:
            frame_filename = os.path.join('frames', username, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f'Frame {frame_count} saved as {frame_filename}')

        if frame_count >=50:
            break
    
    # Release the capture
    cap.release()
    cv2.destroyAllWindows()
    call_model_train_script(request)
