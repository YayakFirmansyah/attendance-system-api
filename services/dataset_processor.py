import os
import cv2
import numpy as np
from utils.face_detector import FaceDetector
from utils.face_encoder import FaceNetEncoder
from config import Config

class EnhancedDatasetProcessor:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_encoder = FaceNetEncoder()
        self.stats = {
            'total_images': 0,
            'faces_detected': 0,
            'faces_encoded': 0,
            'students_processed': 0,
            'failed_images': []
        }
        print("✓ Enhanced dataset processor initialized")
        
    def process_dataset(self, dataset_path=None):
        """Process dataset with detailed statistics"""
        if dataset_path is None:
            dataset_path = Config.DATASET_PATH
            
        print(f"Processing dataset from: {dataset_path}")
        print("=" * 50)
        
        encodings = []
        labels = []
        self.stats = {'total_images': 0, 'faces_detected': 0, 'faces_encoded': 0, 'students_processed': 0, 'failed_images': []}
        
        # Get all student folders
        student_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        student_folders.sort()
        
        print(f"Found {len(student_folders)} student folders")
        
        # Process each student folder
        for student_name in student_folders:
            student_path = os.path.join(dataset_path, student_name)
            
            print(f"\nProcessing: {student_name}")
            print("-" * 30)
            
            # Get all image files
            image_files = [f for f in os.listdir(student_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if len(image_files) == 0:
                print(f"  ⚠️  No images found for {student_name}")
                continue
            
            student_encodings = []
            
            # Process each image
            for image_file in image_files:
                image_path = os.path.join(student_path, image_file)
                self.stats['total_images'] += 1
                
                print(f"  📷 Processing: {image_file}")
                
                # Process image
                encoding = self.process_image(image_path, student_name)
                
                if encoding is not None:
                    student_encodings.append(encoding)
                    encodings.append(encoding)
                    labels.append(student_name)
                    self.stats['faces_encoded'] += 1
                    print(f"     ✅ Encoded successfully")
                else:
                    self.stats['failed_images'].append(image_path)
                    print(f"     ❌ Failed to process")
            
            if len(student_encodings) > 0:
                self.stats['students_processed'] += 1
                print(f"  📊 Generated {len(student_encodings)} encodings for {student_name}")
            else:
                print(f"  ⚠️  No valid encodings for {student_name}")
        
        # Print final statistics
        self.print_statistics()
        
        return encodings, labels
    
    def process_image(self, image_path, student_name):
        """Process single image with detailed logging"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"     ❌ Failed to load image")
                return None
            
            print(f"     📐 Image size: {image.shape}")
            
            # Detect faces with MTCNN
            faces = self.face_detector.detect_faces(image)
            self.stats['faces_detected'] += len(faces)
            
            if not faces:
                print(f"     👤 No faces detected")
                return None
            
            if len(faces) > 1:
                print(f"     👥 Multiple faces detected ({len(faces)}), using best one")
            
            # Use the face with highest confidence
            best_face = max(faces, key=lambda x: x['confidence'])
            print(f"     🎯 Face confidence: {best_face['confidence']:.3f}")
            
            # Extract face
            face = self.face_detector.extract_face(image, best_face)
            
            if face is None:
                print(f"     ❌ Failed to extract face")
                return None
            
            print(f"     ✂️  Extracted face size: {face.shape}")
            
            # Generate encoding with FaceNet
            encoding = self.face_encoder.encode_face(face)
            
            if encoding is not None:
                print(f"     🧠 Generated encoding: {encoding.shape}")
                return encoding
            else:
                print(f"     ❌ Failed to generate encoding")
                return None
                
        except Exception as e:
            print(f"     ❌ Error processing image: {e}")
            return None
    
    def print_statistics(self):
        """Print processing statistics"""
        print("\n" + "=" * 50)
        print("DATASET PROCESSING STATISTICS")
        print("=" * 50)
        print(f"📁 Total images processed: {self.stats['total_images']}")
        print(f"👤 Faces detected: {self.stats['faces_detected']}")
        print(f"🧠 Faces encoded: {self.stats['faces_encoded']}")
        print(f"👥 Students processed: {self.stats['students_processed']}")
        
        success_rate = (self.stats['faces_encoded'] / self.stats['total_images']) * 100 if self.stats['total_images'] > 0 else 0
        print(f"✅ Success rate: {success_rate:.1f}%")
        
        if self.stats['failed_images']:
            print(f"\n❌ Failed images ({len(self.stats['failed_images'])}):")
            for failed_img in self.stats['failed_images'][:5]:  # Show first 5
                print(f"   - {failed_img}")
            if len(self.stats['failed_images']) > 5:
                print(f"   ... and {len(self.stats['failed_images']) - 5} more")
        
        print("=" * 50)
    
    def validate_dataset(self, dataset_path=None):
        """Validate dataset structure before processing"""
        if dataset_path is None:
            dataset_path = Config.DATASET_PATH
        
        print(f"Validating dataset structure: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path does not exist: {dataset_path}")
            return False
        
        student_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        
        if len(student_folders) < 2:
            print(f"❌ Need at least 2 student folders, found {len(student_folders)}")
            return False
        
        valid_students = 0
        total_images = 0
        
        for student_name in student_folders:
            student_path = os.path.join(dataset_path, student_name)
            image_files = [f for f in os.listdir(student_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if len(image_files) >= 2:  # At least 2 images per student
                valid_students += 1
                total_images += len(image_files)
                print(f"✅ {student_name}: {len(image_files)} images")
            else:
                print(f"⚠️  {student_name}: {len(image_files)} images (need at least 2)")
        
        print(f"\n📊 Summary:")
        print(f"   Valid students: {valid_students}/{len(student_folders)}")
        print(f"   Total images: {total_images}")
        
        if valid_students < 2:
            print(f"❌ Need at least 2 students with 2+ images each")
            return False
        
        print(f"✅ Dataset validation passed")
        return True