# attendance-api-new/services/attendance_service.py
import cv2
import numpy as np
from datetime import datetime
from utils.face_detector import FaceDetector
from utils.face_encoder import FaceNetEncoder
from utils.face_recognizer import AdvancedFaceRecognizer
from utils.database import db

class AttendanceService:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_encoder = FaceNetEncoder()
        self.face_recognizer = AdvancedFaceRecognizer()
        
        # Try to load existing model
        self.model_loaded = self.face_recognizer.load_model()
        
        print(f"Attendance Service initialized")
        print(f"Model loaded: {self.model_loaded}")
    
    def process_attendance(self, image, class_id=1):
        """Process attendance from image"""
        try:
            print("Processing attendance...")
            
            if not self.model_loaded:
                return {
                    'success': False,
                    'message': 'Face recognition model not trained yet'
                }
            
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            if not faces:
                return {
                    'success': False,
                    'message': 'No faces detected'
                }
            
            results = []
            
            for i, face_detection in enumerate(faces):
                # Extract face
                face = self.face_detector.extract_face(image, face_detection)
                
                if face is None:
                    continue
                
                # Generate encoding
                encoding = self.face_encoder.encode_face(face)
                
                if encoding is None:
                    continue
                
                # Recognize face
                student_name, confidence, probabilities = self.face_recognizer.predict(
                    encoding, return_probabilities=True
                )
                
                result = {
                    'face_id': i,
                    'student_name': student_name,
                    'confidence': float(confidence),
                    'bounding_box': face_detection['box'],
                    'face_confidence': face_detection['confidence']
                }
                
                # Add top 3 predictions
                if probabilities:
                    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                    result['top_predictions'] = [
                        {'name': name, 'confidence': float(conf)} 
                        for name, conf in sorted_probs[:3]
                    ]
                
                results.append(result)
                
                # Record attendance if confident enough
                if student_name and confidence > 0.8:
                    self.record_attendance(student_name, class_id, confidence)
            
            return {
                'success': True,
                'message': f'Processed {len(results)} face(s)',
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Attendance processing error: {e}")
            return {
                'success': False,
                'message': f'Processing error: {str(e)}'
            }
    
    def record_attendance(self, student_name, class_id, confidence):
        """Record attendance in database"""
        try:
            # For now, just log it (you can integrate with actual database)
            timestamp = datetime.now()
            print(f"📝 Attendance recorded:")
            print(f"   Student: {student_name}")
            print(f"   Class: {class_id}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Time: {timestamp}")
            
            # TODO: Save to database
            # db.save_attendance(student_name, class_id, confidence)
            
            return True
            
        except Exception as e:
            print(f"Record attendance error: {e}")
            return False
    
    def get_model_info(self):
        """Get information about loaded model"""
        if not self.model_loaded:
            return None
        
        return self.face_recognizer.get_model_info()