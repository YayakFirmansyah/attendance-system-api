import tensorflow as tf
import numpy as np
import cv2
from config import Config

class FaceNetEncoder:
    def __init__(self):
        self.model = None
        self.image_size = Config.IMAGE_SIZE
        self.load_facenet()
        
    def load_facenet(self):
        """Load FaceNet model"""
        try:
            print("Loading FaceNet model...")
            
            # Try to load keras-facenet
            try:
                from keras_facenet import FaceNet
                self.model = FaceNet()
                self.model_type = "facenet"
                print("✓ FaceNet model loaded successfully")
                return True
            except:
                print("keras-facenet not available, building simple model...")
                return self.build_simple_model()
                
        except Exception as e:
            print(f"✗ FaceNet loading failed: {e}")
            return self.build_simple_model()
    
    def build_simple_model(self):
        """Fallback: Build simple CNN if FaceNet fails"""
        try:
            print("Building simple CNN encoder...")
            
            # Input
            inputs = tf.keras.Input(shape=(*self.image_size, 3))
            
            # CNN layers
            x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            
            x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            
            # Dense layers  
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            # Embedding layer (512 dimensions for compatibility)
            embeddings = tf.keras.layers.Dense(512)(x)
            embeddings = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embeddings)
            
            self.model = tf.keras.Model(inputs, embeddings)
            self.model_type = "simple_cnn"
            
            # Warm up
            dummy = np.zeros((1, *self.image_size, 3))
            self.model.predict(dummy, verbose=0)
            
            print("✓ Simple CNN encoder built successfully")
            return True
            
        except Exception as e:
            print(f"✗ Simple encoder build failed: {e}")
            return False
    
    def preprocess_face(self, face):
        """Preprocess face for FaceNet"""
        try:
            # Resize to target size
            if face.shape[:2] != self.image_size:
                face = cv2.resize(face, self.image_size)
            
            # Convert to float and normalize
            face = face.astype(np.float32)
            
            # Normalize based on model type
            if hasattr(self, 'model_type') and self.model_type == "facenet":
                # FaceNet preprocessing: scale to [-1, 1]
                face = (face / 255.0 - 0.5) * 2.0
            else:
                # Simple CNN: scale to [0, 1]
                face = face / 255.0
            
            return face
            
        except Exception as e:
            print(f"Face preprocessing error: {e}")
            return None
    
    def encode_face(self, face):
        """Generate face encoding"""
        try:
            if self.model is None:
                return None
            
            # Preprocess face
            processed_face = self.preprocess_face(face)
            if processed_face is None:
                return None
            
            # Add batch dimension
            if len(processed_face.shape) == 3:
                face_batch = np.expand_dims(processed_face, axis=0)
            else:
                face_batch = processed_face
            
            # Generate encoding
            if hasattr(self.model, 'embeddings'):
                # Using keras-facenet
                encoding = self.model.embeddings(face_batch)
                if hasattr(encoding, 'numpy'):
                    encoding = encoding.numpy()
            else:
                # Using custom model
                encoding = self.model.predict(face_batch, verbose=0)
            
            # L2 normalize
            encoding = encoding / np.linalg.norm(encoding, axis=1, keepdims=True)
            
            return encoding[0]
            
        except Exception as e:
            print(f"Face encoding error: {e}")
            return None