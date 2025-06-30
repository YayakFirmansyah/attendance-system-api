# app.py - FIXED VERSION
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
from datetime import datetime

from config import Config
from utils.face_detector import FaceDetector
from utils.face_encoder import FaceNetEncoder
from utils.face_recognizer import AdvancedFaceRecognizer
from utils.scalable_face_recognizer import ScalableFaceRecognizer

app = Flask(__name__)
CORS(app)

# Initialize services
print("🚀 Initializing Face Recognition API...")
print("📊 Using: MTCNN + FaceNet + SVM")

face_detector = FaceDetector()
face_encoder = FaceNetEncoder()
face_recognizer = AdvancedFaceRecognizer()
scalable_recognizer = ScalableFaceRecognizer()

print("🔄 Loading verification database...")
scalable_recognizer.load_database()

# Auto-load model if exists
model_path = os.path.join(Config.MODEL_PATH, 'advanced_face_recognizer.pkl')
if os.path.exists(model_path):
    print("🔄 Loading existing model...")
    if face_recognizer.load_model():
        face_recognizer.update_threshold(Config.RECOGNITION_THRESHOLD)
        print("✅ Model loaded successfully")
    else:
        print("❌ Model loading failed")
else:
    print("📝 No existing model found - need to train first")

def decode_base64_image(base64_string):
    """Decode base64 image to OpenCV format"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Base64 decode error: {e}")
        return None

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0 - MTCNN + FaceNet + SVM',
        'model_loaded': face_recognizer.svm_model is not None,
        'database_students': len(scalable_recognizer.student_encodings),
        'config': {
            'recognition_threshold': Config.RECOGNITION_THRESHOLD,
            'face_confidence_threshold': Config.FACE_CONFIDENCE_THRESHOLD
        }
    })

@app.route('/api/verify-face', methods=['POST'])
def verify_face():
    """Main endpoint for face verification - FIXED"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False, 
                'message': 'No image provided'
            }), 400

        # Decode image
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({
                'success': False, 
                'message': 'Invalid image format'
            }), 400

        print(f"🔍 Processing image: {image.shape}")

        # Detect faces
        faces = face_detector.detect_faces(image)
        if not faces:
            return jsonify({
                'success': False,
                'message': 'No faces detected'
            })

        results = []
        
        for i, face_detection in enumerate(faces):
            try:
                # Extract and encode face
                face = face_detector.extract_face(image, face_detection)
                if face is None:
                    continue
                    
                encoding = face_encoder.encode_face(face)
                if encoding is None:
                    continue

                # FIXED: Use verify_face (singular) instead of verify_faces
                student_id, student_name, similarity, top_matches = scalable_recognizer.verify_face(
                    encoding, top_k=5
                )
                
                # Prepare result
                result = {
                    'face_id': int(i),
                    'student_id': str(student_id) if student_id else None,
                    'student_name': str(student_name) if student_name else None,
                    'similarity': float(similarity) if similarity is not None else 0.0,
                    'verified': bool(similarity >= scalable_recognizer.verification_threshold) if similarity is not None else False,
                    'bounding_box': [int(x) for x in face_detection['box']],
                    'mtcnn_confidence': float(face_detection['confidence']),
                    'verification_threshold': float(scalable_recognizer.verification_threshold)
                }
                
                # Add top matches if available
                if top_matches:
                    result['top_matches'] = [
                        {
                            'student_name': match.get('student_name', ''),
                            'similarity': float(match.get('max_similarity', 0.0))
                        }
                        for match in top_matches[:3]
                    ]
                
                results.append(result)
                
                print(f"Face {i}: {student_name or 'Unknown'} ({similarity:.3f})")
                
            except Exception as e:
                print(f"Error processing face {i}: {e}")
                continue

        return jsonify({
            'success': True,
            'results': results,
            'total_faces': len(results),
            'verified_count': sum(1 for r in results if r.get('verified', False)),
            'processing_time': 0  # Could add timing if needed
        })

    except Exception as e:
        print(f"Verification error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'message': f'Verification error: {str(e)}'
        }), 500

@app.route('/api/add-student', methods=['POST'])
def add_student():
    """Add new student to verification database"""
    try:
        data = request.get_json()
        required_fields = ['student_name', 'nim', 'images']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False, 
                    'message': f'Missing required field: {field}'
                }), 400

        if len(data['images']) < 3:
            return jsonify({
                'success': False, 
                'message': 'At least 3 images required'
            }), 400

        # Process images and generate encodings
        encodings = []
        for i, img_base64 in enumerate(data['images']):
            image = decode_base64_image(img_base64)
            if image is None:
                continue
                
            faces = face_detector.detect_faces(image)
            if not faces:
                continue
                
            face = face_detector.extract_face(image, faces[0])
            if face is None:
                continue
                
            encoding = face_encoder.encode_face(face)
            if encoding is not None:
                encodings.append(encoding)

        if len(encodings) < 2:
            return jsonify({
                'success': False,
                'message': f'Only {len(encodings)} valid encodings generated, need at least 2'
            }), 400

        # Add to database
        success = scalable_recognizer.add_student(
            student_id=data['nim'],
            student_name=data['student_name'],
            nim=data['nim'],
            kelas=data.get('class', ''),
            encodings=encodings
        )

        if success:
            scalable_recognizer.save_database()
            return jsonify({
                'success': True,
                'message': 'Student added successfully',
                'student_info': {
                    'name': data['student_name'],
                    'nim': data['nim'],
                    'encoding_count': len(encodings)
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to add student'
            }), 500

    except Exception as e:
        print(f"Error adding student: {e}")
        return jsonify({
            'success': False,
            'message': f'Error adding student: {str(e)}'
        }), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train model from dataset (fallback method)"""
    try:
        from services.dataset_processor import EnhancedDatasetProcessor
        
        dataset_processor = EnhancedDatasetProcessor()
        
        if not dataset_processor.validate_dataset():
            return jsonify({
                'success': False,
                'message': 'Dataset validation failed'
            }), 400

        encodings, labels = dataset_processor.process_dataset()
        
        if len(encodings) == 0:
            return jsonify({
                'success': False,
                'message': 'No faces found in dataset'
            })

        success = face_recognizer.train(encodings, labels)
        
        if success:
            face_recognizer.update_threshold(Config.RECOGNITION_THRESHOLD)
            face_recognizer.save_model()
            
            return jsonify({
                'success': True,
                'message': f'Model trained with {len(encodings)} encodings',
                'statistics': {
                    'total_encodings': len(encodings),
                    'unique_students': len(set(labels)),
                    'students': list(set(labels))
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Training failed'
            })

    except Exception as e:
        print(f"Training error: {e}")
        return jsonify({
            'success': False,
            'message': f'Training error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 FACE RECOGNITION API v2.0 - FIXED")
    print("📊 Components: MTCNN + FaceNet + SVM")
    print("="*60)
    print(f"📁 Dataset path: {Config.DATASET_PATH}")
    print(f"🎯 Recognition threshold: {Config.RECOGNITION_THRESHOLD}")
    print(f"👥 Students in database: {len(scalable_recognizer.student_encodings)}")
    print("\n📡 Available endpoints:")
    print("- GET  /api/health")
    print("- POST /api/verify-face")
    print("- POST /api/add-student") 
    print("- POST /api/train")
    print("="*60)
    
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)