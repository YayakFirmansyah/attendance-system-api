# utils/face_detector.py - MTCNN SIMPLE
import cv2
import numpy as np
from mtcnn import MTCNN

class FaceDetector:
    def __init__(self, min_face_size=40, confidence_threshold=0.9):
        self.detector = MTCNN(min_face_size=min_face_size)
        self.confidence_threshold = confidence_threshold
        print(f"✓ MTCNN detector initialized (min_size={min_face_size}, threshold={confidence_threshold})")
    
    def detect_faces(self, image):
        """
        Detect faces in image menggunakan MTCNN
        Returns: list of face crops dengan koordinat
        """
        try:
            # Convert to RGB jika BGR
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Detect faces
            results = self.detector.detect_faces(rgb_image)
            
            faces = []
            for result in results:
                confidence = result['confidence']
                
                # Filter by confidence
                if confidence >= self.confidence_threshold:
                    # Get bounding box
                    x, y, w, h = result['box']
                    
                    # Pastikan koordinat valid
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)
                    
                    # Crop face
                    face_crop = rgb_image[y:y+h, x:x+w]
                    
                    if face_crop.size > 0:  # Pastikan crop valid
                        faces.append({
                            'face': face_crop,
                            'box': (x, y, w, h),
                            'confidence': confidence,
                            'keypoints': result['keypoints']
                        })
            
            return faces
            
        except Exception as e:
            print(f"❌ Face detection error: {e}")
            return []
    
    def detect_single_face(self, image):
        """Detect hanya 1 face terbesar/confidence tertinggi"""
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        # Ambil face dengan confidence tertinggi
        best_face = max(faces, key=lambda x: x['confidence'])
        return best_face
    
    def preprocess_face(self, face_crop, target_size=(160, 160)):
        """
        Preprocess face untuk FaceNet input
        FaceNet expects (160, 160, 3) RGB image dengan nilai 0-255
        """
        try:
            # Resize ke target size
            face_resized = cv2.resize(face_crop, target_size)
            
            # Pastikan RGB dan range 0-255
            if face_resized.dtype != np.uint8:
                face_resized = face_resized.astype(np.uint8)
            
            return face_resized
            
        except Exception as e:
            print(f"❌ Face preprocessing error: {e}")
            return None