import cv2
import numpy as np
from mtcnn import MTCNN
from config import Config

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()
        self.confidence_threshold = Config.FACE_CONFIDENCE_THRESHOLD
        print("✓ MTCNN detector initialized")
        
    def detect_faces(self, image):
        """Detect faces in image"""
        try:
            # Convert to RGB
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
                
            # Detect faces
            detections = self.detector.detect_faces(rgb_image)
            
            # Filter by confidence
            valid_faces = []
            for detection in detections:
                if detection['confidence'] >= self.confidence_threshold:
                    valid_faces.append(detection)
                    
            return valid_faces
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def extract_face(self, image, detection, target_size=None):
        """Extract face from detection"""
        if target_size is None:
            target_size = Config.IMAGE_SIZE
            
        try:
            x, y, width, height = detection['box']
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = min(image.shape[1] - x, width + 2 * padding)
            height = min(image.shape[0] - y, height + 2 * padding)
            
            # Extract face
            face = image[y:y+height, x:x+width]
            
            # Resize
            face_resized = cv2.resize(face, target_size)
            
            return face_resized
            
        except Exception as e:
            print(f"Face extraction error: {e}")
            return None