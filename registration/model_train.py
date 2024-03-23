import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import datetime

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = os.path.join(dir, im_name)
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)

    def plot_images(self):
        plt.figure(figsize=(18,16))
        for num,image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y)//ncols + 1
            plt.subplot(nrows,ncols,num+1)
            plt.imshow(image)
            plt.axis('off')

def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)
    # 4D (Nonex160x160x3)
    yhat= embedder.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)

#Here chipi chipi chapa chapa

# Directory containing the face images
directory = "frames"

# Initialize FACELOADING class
faceloading = FACELOADING(directory)
X, Y = faceloading.load_classes()

# Initialize FaceNet embedder
embedder = FaceNet()

# Generate embeddings for each face image
EMBEDDED_X = [get_embedding(img) for img in X]
EMBEDDED_X = np.asarray(EMBEDDED_X)

# Save embeddings and labels with versioning
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
embeddings_filename = f'templates/models/faces_embeddings_done_4classes_{timestamp}.npz'
np.savez_compressed(embeddings_filename, EMBEDDED_X, Y)

# Encode labels
encoder = LabelEncoder()
encoder.fit(Y)
Y_encoded = encoder.transform(Y)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y_encoded, shuffle=True, random_state=17)

# Initialize and train SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

# Predictions
ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(Y_train, ypreds_train)

# Save the model with versioning
model_filename = f'templates/models/svm_model_160x160_{timestamp}.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

# Print training accuracy
print("Training Accuracy:", train_accuracy)