# utils/face_encoder.py - FaceNet SIMPLE
import numpy as np
from keras_facenet import FaceNet

class FaceEncoder:
    def __init__(self):
        self.embedder = FaceNet()
        print("✓ FaceNet encoder initialized")
    
    def encode_face(self, face_image):
        """
        Convert face image menjadi 512-dimensional embedding
        face_image: RGB image (160, 160, 3) atau akan di-resize otomatis
        """
        try:
            # Pastikan input adalah numpy array
            if not isinstance(face_image, np.ndarray):
                face_image = np.array(face_image)
            
            # Pastikan shape benar
            if len(face_image.shape) == 3:
                # Resize ke 160x160 jika perlu
                if face_image.shape[:2] != (160, 160):
                    import cv2
                    face_image = cv2.resize(face_image, (160, 160))
                
                # Expand dimensions untuk batch
                face_batch = np.expand_dims(face_image, axis=0)
            else:
                print(f"❌ Invalid face image shape: {face_image.shape}")
                return None
            
            # Generate embedding
            embedding = self.embedder.embeddings(face_batch)
            
            # Return as 1D array
            return embedding[0]  # Shape: (512,)
            
        except Exception as e:
            print(f"❌ Face encoding error: {e}")
            return None
    
    def encode_faces_batch(self, face_images):
        """
        Encode multiple faces sekaligus (lebih efisien)
        """
        try:
            if not face_images:
                return []
            
            # Prepare batch
            batch = []
            for face in face_images:
                if not isinstance(face, np.ndarray):
                    face = np.array(face)
                
                # Resize jika perlu
                if face.shape[:2] != (160, 160):
                    import cv2
                    face = cv2.resize(face, (160, 160))
                
                batch.append(face)
            
            batch = np.array(batch)
            
            # Generate embeddings
            embeddings = self.embedder.embeddings(batch)
            
            return embeddings  # Shape: (N, 512)
            
        except Exception as e:
            print(f"❌ Batch encoding error: {e}")
            return []