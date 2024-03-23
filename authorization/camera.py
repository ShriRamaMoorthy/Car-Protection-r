import cv2
import os
import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
import time
import winsound
import smtplib
import ssl
from email.message import EmailMessage
import pywhatkit
import pyautogui
from pynput.keyboard import Key, Controller
from django.http import HttpResponseRedirect
from django.urls import reverse
import urllib.request  
import webbrowser 


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        print("hello")
        # Initialize FaceNet
        self.facenet = FaceNet()

        self.load_model()
        # Load face embeddings and labels
        # self.faces_embeddings = np.load("templates/models/faces_embeddings_done_srm_kis.npz")
        self.Y = self.faces_embeddings['arr_1']

        # Encode class labels
        self.encoder = LabelEncoder()
        self.encoder.fit(self.Y)
        self.first_unknown_detected = False

        # Load Haar Cascade classifier for face detection
        self.haarcascade = cv2.CascadeClassifier("templates/models/haarcascade_frontalface_default.xml")

        # Load SVM model for face recognition
        # self.model = pickle.load(open("templates/models/svm_model_160x160_srm_kish.pkl", 'rb'))

        # Initialize last beep time
        self.last_beep_time = time.time()

        self.identity_count=0
        self.redirect_needed=False
    
    def load_model(self):
        # Directory containing the model files
        directory = 'templates/models'

        # Function to parse timestamp from filename
        def parse_timestamp(filename):
            timestamp_str = filename.split('_')[-2] + '_' + filename.split('_')[-1].split('.')[0]
            return datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

        # Get current timestamp
        current_timestamp = datetime.datetime.now()

        # List all files in the directory
        files = os.listdir(directory)

        # Filter out only the model files
        model_files = [file for file in files if file.startswith('faces_embeddings_done_4classes_') and file.endswith('.npz')]

        # Find the model file with the closest timestamp
        closest_file = None
        closest_difference = float('inf')

        for file in model_files:
            timestamp = parse_timestamp(file)
            time_difference = abs(current_timestamp - timestamp).total_seconds()
            if time_difference < closest_difference:
                closest_file = file
                closest_difference = time_difference

        # Load the closest model
        if closest_file:
            embeddings_filename = os.path.join(directory, closest_file)
            self.faces_embeddings = np.load(embeddings_filename)
            print(f"Loaded model from: {embeddings_filename}")
        else:
            print("No model files found.")

        model_files = [file for file in files if file.startswith('svm_model_160x160_') and file.endswith('.pkl')]

        # Find the model file with the closest timestamp
        closest_file = None
        closest_difference = float('inf')

        for file in model_files:
            timestamp = parse_timestamp(file)
            time_difference = abs(current_timestamp - timestamp).total_seconds()
            if time_difference < closest_difference:
                closest_file = file
                closest_difference = time_difference

        # Load the closest model
        if closest_file:
            embeddings_filename = os.path.join(directory, closest_file)
            self.model = self.model = pickle.load(open(embeddings_filename, 'rb'))
            print(f"Loaded model from: {embeddings_filename}")
        else:
            print("No model files found.")


    def __del__(self):
        self.video.release()

    def beep(self):
        # Implement your beep sound function here
        print("Beep sound triggered")
        self.send_notifications()
        for _ in range(5):  # Repeat the beep sound 5 times
            winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 milliseconds

    def send_notifications(self):
        # Define email sender and receiver
        email_sender = "2032020mds@cit.edu.in"
        email_password = 'eytl qcnq yvdu wugn'
        email_receiver = "dhanushkumar183@gmail.com"

        # Set the subject and body of the email
        subject = 'Check out my new video!'
        body = """
        a1c8-2409-40f4-1019-8bc1-74b7-ca1e-4596-7e71.ngrok-free.app/monitor/
        """

        em = EmailMessage()
        em['From'] = email_sender
        em['To'] = email_receiver
        em['Subject'] = subject
        em.set_content(body)

        # Add SSL (layer of security)
        context = ssl.create_default_context()

        # Log in and send the email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(email_sender, email_password)
            smtp.sendmail(email_sender, email_receiver, em.as_string())

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = self.haarcascade.detectMultiScale(gray_img, 1.3, 5)
        print("Get frame")

        for x, y, w, h in faces:
            img = rgb_img[y:y+h, x:x+w]
            img = cv2.resize(img, (160, 160))  # Resize face to 160x160 pixels
            img = np.expand_dims(img, axis=0)
            ypred = self.facenet.embeddings(img)  # Get face embeddings
            class_probabilities = self.model.predict_proba(ypred)  # Predict class probabilities
            max_probability = np.max(class_probabilities)
            if max_probability < 0.8:  # Set your threshold here
                final_name = "Unknown"
                print("Unknown...............")
                if not self.first_unknown_detected:  # Check if it's the first time an unknown face is detected
                    self.beep()  # Call beep function the first time an unknown face is detected
                    self.first_unknown_detected = True  # Update flag to indicate that first unknown is detected

                # Check if 5 minutes have passed since the last beep
                elif time.time() - self.last_beep_time >= 300:
                    self.beep()
                    self.last_beep_time = time.time()

            else:
                face_name = np.argmax(class_probabilities)
                final_name = self.encoder.inverse_transform([face_name])[0]  # Convert label to name

            # Draw a rectangle around the detected face and display the name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
            cv2.putText(frame, str(final_name), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3, cv2.LINE_AA)
            if(final_name!="Unknown"):
                self.identity_count+=1
                print(self.identity_count)
                if(self.identity_count>=15):
                    self.redirect_needed=True
                    print("you are successfully verified")
                    url ='http://127.0.0.1:8000/drowsy'
                    webbrowser.open_new_tab(url)
                    return None

        # Encode the frame in JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

