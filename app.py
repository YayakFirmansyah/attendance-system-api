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
from services.dataset_processor import EnhancedDatasetProcessor

app = Flask(__name__)
CORS(app)

# Initialize services
print("Initializing Face Recognition API...")
print("Using: MTCNN + FaceNet + Advanced SVM")

face_detector = FaceDetector()
face_encoder = FaceNetEncoder()
face_recognizer = AdvancedFaceRecognizer()
dataset_processor = EnhancedDatasetProcessor()

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
    """Health check with detailed service status"""
    model_info = face_recognizer.get_model_info()
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0 - MTCNN + FaceNet + SVM',
        'services': {
            'face_detection': face_detector.detector is not None,
            'face_encoding': face_encoder.model is not None,
            'face_recognition': face_recognizer.svm_model is not None,
        },
        'model_info': model_info,
        'dataset_path': Config.DATASET_PATH
    })

@app.route('/api/validate-dataset', methods=['GET'])
def validate_dataset():
    """Validate dataset structure before training"""
    try:
        is_valid = dataset_processor.validate_dataset()
        
        return jsonify({
            'success': is_valid,
            'message': 'Dataset validation completed',
            'dataset_path': Config.DATASET_PATH
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Validation error: {str(e)}'
        }), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train model from dataset with enhanced processing"""
    try:
        data = request.get_json() if request.is_json else {}
        optimize_hyperparams = data.get('optimize_hyperparams', True)
        
        print("🚀 Starting enhanced training process...")
        print("Components: MTCNN + FaceNet + Advanced SVM")
        
        # Validate dataset first
        if not dataset_processor.validate_dataset():
            return jsonify({
                'success': False,
                'message': 'Dataset validation failed'
            }), 400
        
        # Process dataset
        print("\n📊 Processing dataset...")
        encodings, labels = dataset_processor.process_dataset()
        
        if len(encodings) == 0:
            return jsonify({
                'success': False,
                'message': 'No faces found in dataset'
            })
        
        if len(set(labels)) < 2:
            return jsonify({
                'success': False,
                'message': f'Need at least 2 different students, found {len(set(labels))}'
            })
        
        # Train advanced SVM
        print("\n🤖 Training Advanced SVM...")
        success = face_recognizer.train(encodings, labels, optimize_hyperparams)
        
        if success:
            model_info = face_recognizer.get_model_info()
            
            return jsonify({
                'success': True,
                'message': f'Model trained successfully with {len(encodings)} face encodings',
                'statistics': {
                    'total_encodings': len(encodings),
                    'unique_students': len(set(labels)),
                    'students': list(set(labels)),
                    'encodings_per_student': {
                        label: labels.count(label) for label in set(labels)
                    }
                },
                'model_info': model_info,
                'processing_stats': dataset_processor.stats
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Training failed'
            })
            
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Training error: {str(e)}'
        }), 500

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Recognize face with detailed analysis"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
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
        
        # Detect faces with MTCNN
        faces = face_detector.detect_faces(image)
        
        if not faces:
            return jsonify({
                'success': False,
                'message': 'No faces detected by MTCNN'
            })
        
        print(f"👥 Detected {len(faces)} face(s)")
        
        results = []
        
        for i, face_detection in enumerate(faces):
            print(f"\n📸 Processing face {i+1}")
            print(f"   MTCNN confidence: {face_detection['confidence']:.3f}")
            
            # Extract face
            face = face_detector.extract_face(image, face_detection)
            
            if face is None:
                print(f"   ❌ Failed to extract face")
                continue
            
            print(f"   ✂️  Extracted face: {face.shape}")
            
            # Generate encoding with FaceNet
            encoding = face_encoder.encode_face(face)
            
            if encoding is None:
                print(f"   ❌ Failed to generate FaceNet encoding")
                continue
            
            print(f"   🧠 FaceNet encoding: {encoding.shape}")
            
            # Recognize with Advanced SVM
            student_name, confidence, probabilities = face_recognizer.predict(
                encoding, return_probabilities=True
            )
            
            print(f"   🎯 Recognition result: {student_name} ({confidence:.3f})")
            
            result = {
                'face_id': i,
                'student_name': student_name,
                'confidence': float(confidence),
                'bounding_box': face_detection['box'],
                'mtcnn_confidence': float(face_detection['confidence']),
                'encoding_shape': encoding.shape
            }
            
            # Add top predictions if available
            if probabilities:
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                result['top_predictions'] = [
                    {'name': name, 'confidence': float(conf)} 
                    for name, conf in sorted_probs[:3]
                ]
            
            results.append(result)
        
        return jsonify({
            'success': True,
            'message': f'Processed {len(results)} face(s)',
            'results': results,
            'processing_info': {
                'total_faces_detected': len(faces),
                'faces_processed': len(results),
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        print(f"Recognition error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Recognition error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 FACE RECOGNITION API v2.0")
    print("📊 Components: MTCNN + FaceNet + Advanced SVM")
    print("="*60)
    print(f"📁 Dataset path: {Config.DATASET_PATH}")
    print(f"🔧 Face confidence threshold: {Config.FACE_CONFIDENCE_THRESHOLD}")
    print(f"🎯 Recognition threshold: {Config.RECOGNITION_THRESHOLD}")
    print("\n📡 Available endpoints:")
    print("- GET  /api/health")
    print("- GET  /api/validate-dataset")
    print("- POST /api/train")
    print("- POST /api/recognize")
    print("="*60)
    
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)