import tensorflow as tf
import numpy as np
import cv2
import random
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
                print("keras-facenet not available, building enhanced model...")
                return self.build_enhanced_model()
                
        except Exception as e:
            print(f"✗ FaceNet loading failed: {e}")
            return self.build_enhanced_model()
    
    def build_enhanced_model(self):
        """Build enhanced CNN with better architecture"""
        try:
            print("Building enhanced CNN encoder...")
            
            # Input dengan normalisasi
            inputs = tf.keras.Input(shape=(*self.image_size, 3))
            
            # Preprocessing layer
            x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)(inputs)
            x = tf.keras.layers.Lambda(lambda x: (x - 0.5) * 2.0)(x)  # Normalize to [-1, 1]
            
            # Enhanced CNN architecture
            # Block 1
            x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = tf.keras.layers.Dropout(0.25)(x)
            
            # Block 2
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = tf.keras.layers.Dropout(0.25)(x)
            
            # Block 3
            x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = tf.keras.layers.Dropout(0.25)(x)
            
            # Block 4
            x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            
            # Dense layers dengan dropout
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            # Embedding layer (512 dimensions)
            embeddings = tf.keras.layers.Dense(512)(x)
            # L2 normalization yang lebih stabil
            embeddings = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1, epsilon=1e-12))(embeddings)
            
            self.model = tf.keras.Model(inputs, embeddings)
            self.model_type = "enhanced_cnn"
            
            # Compile model untuk stabilitas
            self.model.compile(optimizer='adam', loss='mse')
            
            # Warm up dengan beberapa dummy input
            print("Warming up model...")
            for i in range(3):
                dummy = np.random.randint(0, 255, (1, *self.image_size, 3), dtype=np.uint8)
                _ = self.model.predict(dummy, verbose=0)
            
            print("✓ Enhanced CNN encoder built successfully")
            return True
            
        except Exception as e:
            print(f"✗ Enhanced encoder build failed: {e}")
            return False
    
    def enhanced_preprocess_face(self, face):
        """Enhanced preprocessing dengan multiple techniques"""
        try:
            # Pastikan face adalah BGR (OpenCV format)
            if len(face.shape) != 3 or face.shape[2] != 3:
                print(f"Invalid face shape: {face.shape}")
                return None
            
            # 1. Resize dengan interpolasi yang baik
            if face.shape[:2] != self.image_size:
                face = cv2.resize(face, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            
            # 2. Perbaikan pencahayaan (Histogram Equalization)
            # Convert ke YUV dan equalize channel Y (luminance)
            yuv = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            face = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            # 3. Noise reduction dengan Gaussian blur ringan
            face = cv2.GaussianBlur(face, (3, 3), 0)
            
            # 4. Peningkatan kontras menggunakan CLAHE
            lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 5. Pastikan dalam range yang benar
            face = np.clip(face, 0, 255)
            face = face.astype(np.uint8)
            
            return face
            
        except Exception as e:
            print(f"Enhanced preprocessing error: {e}")
            return None
    
    def encode_single_face(self, face):
        """Encode satu wajah tanpa augmentasi untuk konsistensi"""
        try:
            if self.model is None:
                return None
            
            # Preprocess face
            processed_face = self.enhanced_preprocess_face(face)
            if processed_face is None:
                return None
            
            # Add batch dimension
            face_batch = np.expand_dims(processed_face, axis=0)
            
            # Generate encoding
            if hasattr(self.model, 'embeddings'):
                # Using keras-facenet
                encoding = self.model.embeddings(face_batch)
                if hasattr(encoding, 'numpy'):
                    encoding = encoding.numpy()
            else:
                # Using custom model
                encoding = self.model.predict(face_batch, verbose=0)
            
            # Pastikan encoding dalam bentuk yang benar
            if len(encoding.shape) > 1:
                encoding = encoding[0]  # Ambil batch pertama
            
            # Validasi encoding
            if np.any(np.isnan(encoding)) or np.any(np.isinf(encoding)):
                print("Warning: Invalid encoding detected (NaN/inf)")
                return None
            
            # Final L2 normalization
            norm = np.linalg.norm(encoding)
            if norm > 0:
                encoding = encoding / norm
            else:
                print("Warning: Zero norm encoding")
                return None
            
            return encoding
            
        except Exception as e:
            print(f"Single face encoding error: {e}")
            return None
    
    def encode_face(self, face):
        """Main encoding function - KONSISTEN untuk training dan testing"""
        # PERBAIKAN: Gunakan encoding tunggal yang konsisten
        # Tidak menggunakan augmentasi yang bisa menyebabkan hasil berbeda
        return self.encode_single_face(face)
    
    def encode_face_with_augmentation(self, face, num_augmentations=2):
        """Optional: Generate robust encoding dengan augmentasi (untuk training khusus)"""
        try:
            encodings = []
            
            # 1. Encoding original
            original_encoding = self.encode_single_face(face)
            if original_encoding is not None:
                encodings.append(original_encoding)
            
            # 2. Generate augmented encodings
            for i in range(num_augmentations):
                augmented_face = self.apply_augmentation(face, i)
                if augmented_face is not None:
                    aug_encoding = self.encode_single_face(augmented_face)
                    if aug_encoding is not None:
                        encodings.append(aug_encoding)
            
            if len(encodings) == 0:
                return None
            
            # 3. Rata-ratakan encodings untuk stabilitas
            avg_encoding = np.mean(encodings, axis=0)
            
            # 4. Renormalize
            norm = np.linalg.norm(avg_encoding)
            if norm > 0:
                avg_encoding = avg_encoding / norm
            
            return avg_encoding
            
        except Exception as e:
            print(f"Robust encoding error: {e}")
            return None
    
    def apply_augmentation(self, face, aug_type):
        """Apply augmentasi ringan untuk variasi"""
        try:
            # Set random seed untuk hasil yang reproducible
            np.random.seed(42 + aug_type)
            random.seed(42 + aug_type)
            
            if aug_type == 0:
                # Sedikit rotasi (-2 sampai 2 derajat - lebih kecil)
                angle = random.uniform(-2, 2)
                center = (face.shape[1]//2, face.shape[0]//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                return cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))
            
            elif aug_type == 1:
                # Sedikit perubahan brightness
                hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
                brightness_factor = random.uniform(0.95, 1.05)
                hsv[:,:,2] = np.clip(hsv[:,:,2] * brightness_factor, 0, 255)
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            elif aug_type == 2:
                # Sedikit perubahan kontras
                alpha = random.uniform(0.98, 1.02)  # Lebih kecil
                beta = random.uniform(-2, 2)        # Lebih kecil
                return cv2.convertScaleAbs(face, alpha=alpha, beta=beta)
            
            return face
            
        except Exception as e:
            print(f"Augmentation error: {e}")
            return face
    
    def preprocess_face(self, face):
        """Backward compatibility - redirect ke enhanced preprocessing"""
        return self.enhanced_preprocess_face(face)
    
    def validate_encoding(self, encoding):
        """Validasi kualitas encoding"""
        if encoding is None:
            return False, "Encoding is None"
        
        if not isinstance(encoding, np.ndarray):
            return False, "Encoding bukan numpy array"
        
        if encoding.shape[0] != 512:
            return False, f"Encoding dimension salah: {encoding.shape[0]} (expected 512)"
        
        if np.any(np.isnan(encoding)):
            return False, "Encoding mengandung NaN"
        
        if np.any(np.isinf(encoding)):
            return False, "Encoding mengandung infinity"
        
        norm = np.linalg.norm(encoding)
        if norm < 0.1:
            return False, f"Encoding norm terlalu kecil: {norm}"
        
        return True, "Encoding valid"
    
    def debug_encoding_process(self, face):
        """Debug fungsi untuk melihat proses encoding step by step"""
        print(f"\n🔧 DEBUGGING ENCODING PROCESS")
        print(f"Input face shape: {face.shape}")
        
        # Step 1: Preprocessing
        processed = self.enhanced_preprocess_face(face)
        if processed is None:
            print("❌ Preprocessing failed")
            return None
        print(f"✅ Preprocessing OK: {processed.shape}")
        
        # Step 2: Model prediction
        try:
            face_batch = np.expand_dims(processed, axis=0)
            print(f"Batch shape: {face_batch.shape}")
            
            encoding = self.model.predict(face_batch, verbose=0)
            print(f"Raw encoding shape: {encoding.shape}")
            print(f"Raw encoding sample: {encoding[0][:5]}...")
            
            # Step 3: Normalization
            if len(encoding.shape) > 1:
                encoding = encoding[0]
            
            norm_before = np.linalg.norm(encoding)
            print(f"Norm before final normalization: {norm_before}")
            
            if norm_before > 0:
                encoding = encoding / norm_before
                norm_after = np.linalg.norm(encoding)
                print(f"Norm after final normalization: {norm_after}")
            
            # Step 4: Validation
            is_valid, msg = self.validate_encoding(encoding)
            print(f"Validation: {'✅' if is_valid else '❌'} {msg}")
            
            return encoding if is_valid else None
            
        except Exception as e:
            print(f"❌ Model prediction failed: {e}")
            return None