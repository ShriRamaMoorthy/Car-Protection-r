import cv2
import torch
from PIL import Image
import torchvision.transforms
import timm
from ultralytics import YOLO

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0.
        self.video = cv2.VideoCapture(0)
        # Load YOLOv8 model for object detection
        self.model_yolo = YOLO('templates/models/best.pt')
        # Load pre-trained MobileViT-V2 model with a custom classifier for gender detection
        self.gender_model = timm.create_model('mobilevitv2_075.cvnets_in1k', pretrained=True, num_classes=2, global_pool='catavgmax')
        state_dict = torch.load('templates/models/gender_srm.pt', map_location=torch.device('cpu'))
        self.gender_model.load_state_dict(state_dict)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        ret, frame = self.video.read()
        
        # Perform object detection
        results = self.model_yolo(frame)
        
        for result in results:
            # Extract bounding box information
            boxes = result.boxes.xyxy   # box with xyxy format, (N, 4)
            confidences = result.boxes.conf   # confidence score, (N, 1)
            classes = result.boxes.cls    # cls, (N, 1)

            for box, confidence, obj_class in zip(boxes, confidences, classes):
                confidence = float(confidence)
                
                
                x_min, y_min, x_max, y_max = box.tolist()
                
                # Draw bounding box and label
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(frame, f"{self.model_yolo.names[int(obj_class)]}: {confidence:.2f}", (int(x_min), int(y_min) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Perform gender detection on the entire frame
        with torch.no_grad():
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_pil = frame_pil.resize((256, 256))  # Assuming MobileViT-V2 requires input size 224x224
                    
            # Convert PIL image to tensor and normalize
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),  
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            frame_tensor = transform(frame_pil).unsqueeze(0)  # Add batch dimension
            
            # Perform gender detection
            gender_output = self.gender_model(frame_tensor)
            gender_prob = torch.softmax(gender_output, dim=1)[0]  # Apply softmax and get probabilities
            pred_gender = torch.argmax(gender_prob).item()  # Get predicted gender (0 for male, 1 for female)
            
            # Display gender prediction
            gender_label = "Male" if pred_gender == 0 else "Female"
            cv2.putText(frame, f"Gender: {gender_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Encode the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def gen(camera):
    while True:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + camera.get_frame() + b'\r\n\r\n')

