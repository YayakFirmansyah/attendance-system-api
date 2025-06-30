# utils/face_detector.py - Fixed and Simplified
import cv2
import numpy as np
from mtcnn import MTCNN
from config import Config

class FaceDetector:
    def __init__(self):
        try:
            print("🔄 Initializing MTCNN detector...")
            self.detector = MTCNN()
            self.confidence_threshold = Config.FACE_CONFIDENCE_THRESHOLD
            print("✅ MTCNN detector initialized")
        except Exception as e:
            print(f"❌ MTCNN initialization failed: {e}")
            self.detector = None
        
    def detect_faces(self, image):
        """Detect faces in image"""
        try:
            if self.detector is None:
                return []
                
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
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
            
            # Add some padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = min(image.shape[1] - x, width + 2 * padding)
            height = min(image.shape[0] - y, height + 2 * padding)
            
            # Extract face region
            face = image[y:y+height, x:x+width]
            
            # Validate face
            if face.size == 0:
                return None
                
            # Resize to target size
            face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
            
            return face_resized
            
        except Exception as e:
            print(f"Face extraction error: {e}")
            return None
    
    def get_face_info(self, detection):
        """Get information about detected face"""
        return {
            'box': detection['box'],
            'confidence': detection['confidence'],
            'keypoints': detection.get('keypoints', {})
        }