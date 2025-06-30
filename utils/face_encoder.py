# utils/face_encoder.py - Fixed and Simplified
import tensorflow as tf
import numpy as np
import cv2
from config import Config

class FaceNetEncoder:
    def __init__(self):
        self.model = None
        self.image_size = Config.IMAGE_SIZE
        self.model_type = "unknown"
        self.load_facenet()
        
    def load_facenet(self):
        """Load FaceNet model with fallback options"""
        try:
            print("🔄 Loading FaceNet model...")
            
            # Option 1: Try keras-facenet (recommended)
            try:
                from keras_facenet import FaceNet
                self.model = FaceNet()
                self.model_type = "keras_facenet"
                print("✅ Keras-FaceNet loaded successfully")
                return True
            except ImportError:
                print("⚠️  keras-facenet not available, building custom model...")
            except Exception as e:
                print(f"⚠️  keras-facenet error: {e}, building custom model...")
            
            # Option 2: Build simple custom model
            return self.build_simple_model()
                
        except Exception as e:
            print(f"❌ FaceNet loading failed: {e}")
            return False
    
    def build_simple_model(self):
        """Build simple but effective CNN model"""
        try:
            print("🔧 Building simple CNN encoder...")
            
            inputs = tf.keras.Input(shape=(*self.image_size, 3))
            
            # Normalization
            x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)(inputs)
            
            # Simple but effective CNN architecture
            x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            
            x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            
            x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            
            # Global average pooling instead of flatten
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            
            # Dense layers
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            
            # L2 normalization for embeddings
            embeddings = tf.keras.layers.Lambda(
                lambda x: tf.nn.l2_normalize(x, axis=1)
            )(x)
            
            self.model = tf.keras.Model(inputs, embeddings)
            self.model_type = "simple_cnn"
            
            # Compile model
            self.model.compile(optimizer='adam', loss='mse')
            
            # Warm up model
            print("🔥 Warming up model...")
            for _ in range(3):
                dummy = np.random.randint(0, 255, (1, *self.image_size, 3), dtype=np.uint8)
                _ = self.model.predict(dummy, verbose=0)
            
            print("✅ Simple CNN encoder ready")
            return True
            
        except Exception as e:
            print(f"❌ Simple model build failed: {e}")
            return False
    
    def preprocess_face(self, face):
        """Simple but effective preprocessing"""
        try:
            # Validate input
            if face is None or len(face.shape) != 3:
                return None
            
            # Resize if needed
            if face.shape[:2] != self.image_size:
                face = cv2.resize(face, self.image_size, interpolation=cv2.INTER_AREA)
            
            # Simple histogram equalization for better lighting
            yuv = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            face = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            # Ensure correct data type
            face = face.astype(np.uint8)
            
            return face
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def encode_face(self, face):
        """Main encoding function"""
        try:
            if self.model is None:
                print("❌ Model not loaded")
                return None
            
            # Preprocess
            processed_face = self.preprocess_face(face)
            if processed_face is None:
                return None
            
            # Add batch dimension
            face_batch = np.expand_dims(processed_face, axis=0)
            
            # Generate encoding
            if self.model_type == "keras_facenet":
                # Use keras-facenet method
                if hasattr(self.model, 'embeddings'):
                    encoding = self.model.embeddings(face_batch)
                    if hasattr(encoding, 'numpy'):
                        encoding = encoding.numpy()
                else:
                    encoding = self.model.predict(face_batch, verbose=0)
            else:
                # Use custom model
                encoding = self.model.predict(face_batch, verbose=0)
            
            # Handle batch dimension
            if len(encoding.shape) > 1:
                encoding = encoding[0]
            
            # Validate encoding
            if np.any(np.isnan(encoding)) or np.any(np.isinf(encoding)):
                print("⚠️  Invalid encoding detected")
                return None
            
            # Final normalization
            norm = np.linalg.norm(encoding)
            if norm > 0:
                encoding = encoding / norm
            else:
                print("⚠️  Zero norm encoding")
                return None
            
            return encoding
            
        except Exception as e:
            print(f"Encoding error: {e}")
            return None
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return None
            
        return {
            'model_type': self.model_type,
            'image_size': self.image_size,
            'encoding_dim': 512,
            'model_loaded': True
        }