# app.py - SIMPLE VERSION (MTCNN + FaceNet + SVM) + GPU SUPPORT
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
from datetime import datetime

# Konfigurasi GPU harus dipanggil SEBELUM import TensorFlow models
from gpu_config import configure_gpu
print("🎮 Mengkonfigurasi GPU...")
configure_gpu()

# Import utils
from utils.face_detector import FaceDetector
from utils.face_encoder import FaceEncoder
from utils.model_loader import ModelLoader

app = Flask(__name__)
CORS(app)

# Initialize components
print("🚀 Initializing Face Recognition System...")
print("📊 Components: MTCNN + FaceNet + SVM")

# Load components
face_detector = FaceDetector(min_face_size=40, confidence_threshold=0.9)
face_encoder = FaceEncoder()
model_loader = ModelLoader(model_dir="models")

# Load trained models
print("\n🔄 Loading trained models from .pkl files...")
if model_loader.load_models():
    print("✅ All models loaded successfully!")
else:
    print("❌ Failed to load models - check models/ folder")

def decode_base64_image(base64_string):
    """Decode base64 image ke OpenCV format"""
    try:
        # Remove data URL prefix jika ada
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
        
    except Exception as e:
        print(f"❌ Base64 decode error: {e}")
        return None

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_info = model_loader.get_model_info()
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': 'Simple MTCNN + FaceNet + SVM',
        'components': {
            'face_detector': 'MTCNN',
            'face_encoder': 'FaceNet (512d)',
            'classifier': 'SVM from main.ipynb'
        },
        'model_info': model_info
    })

@app.route('/api/verify-face', methods=['POST'])
def verify_face():
    """
    Main endpoint untuk face recognition dengan performance timing
    Input: base64 image
    Output: detected faces dengan identity predictions + timing metrics
    """
    try:
        # Validate input
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'message': 'No image provided'
            }), 400
            
        # Extract dynamic thresholds with defaults
        predict_threshold = float(data.get('predict_threshold', 0.15))
        min_confidence = float(data.get('min_confidence', 0.08))
        min_gap = float(data.get('min_gap', 0.03))
        single_class_confidence = float(data.get('single_class_confidence', 0.10))
        
        # Check if models loaded
        if not model_loader.is_loaded:
            return jsonify({
                'success': False,
                'message': 'Models not loaded'
            }), 500
        
        # Decode image
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Invalid image format'
            }), 400
        
        print(f"📷 Processing image: {image.shape}")
        
        # ===== MTCNN FACE DETECTION WITH TIMING =====
        mtcnn_start = cv2.getTickCount()
        faces = face_detector.detect_faces(image)
        mtcnn_end = cv2.getTickCount()
        mtcnn_time = (mtcnn_end - mtcnn_start) / cv2.getTickFrequency() * 1000  # milliseconds
        
        if not faces:
            return jsonify({
                'success': True,
                'message': 'No faces detected',
                'results': [],
                'timing': {
                    'mtcnn_detection_ms': round(mtcnn_time, 2),
                    'facenet_encoding_ms': 0,
                    'svm_classification_ms': 0,
                    'total_processing_ms': round(mtcnn_time, 2)
                }
            })
        
        print(f"👥 Detected {len(faces)} face(s) in {mtcnn_time:.2f}ms")
        
        # Process each face dengan timing individual
        results = []
        total_facenet_time = 0
        total_svm_time = 0
        
        for i, face_data in enumerate(faces):
            face_crop = face_data['face']
            box = face_data['box']
            detection_confidence = face_data['confidence']
            
            # Preprocess face for FaceNet
            processed_face = face_detector.preprocess_face(face_crop)
            if processed_face is None:
                continue
            
            # ===== FACENET ENCODING WITH TIMING =====
            import time
            facenet_start = time.perf_counter()
            embedding = face_encoder.encode_face(processed_face)
            facenet_end = time.perf_counter()
            facenet_time = (facenet_end - facenet_start) * 1000  # milliseconds
            total_facenet_time += facenet_time
            
            if embedding is None:
                continue
            
            # ===== SVM CLASSIFICATION WITH TIMING =====
            svm_start = time.process_time()
            predicted_name, recognition_confidence = model_loader.predict(
                embedding, threshold=predict_threshold  # Threshold dinamis dari request
            )
            # Get all predictions untuk analisis
            all_predictions = model_loader.get_all_predictions(embedding)
            svm_end = time.process_time()
            svm_time = (svm_end - svm_start) * 1000  # milliseconds
            total_svm_time += svm_time
            
            # Enhanced verification logic
            is_verified = False
            if predicted_name != "unknown":
                # Cek confidence gap - apakah prediksi terbaik cukup berbeda dari yang kedua
                sorted_preds = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_preds) >= 2:
                    confidence_gap = sorted_preds[0][1] - sorted_preds[1][1]
                    # Verifikasi jika confidence >= min_confidence DAN gap >= min_gap
                    is_verified = recognition_confidence >= min_confidence and confidence_gap >= min_gap
                else:
                    # Hanya 1 class, threshold dinamis
                    is_verified = recognition_confidence >= single_class_confidence
            
            # Hasil untuk face ini dengan timing per-face
            face_result = {
                'face_id': i + 1,
                'bounding_box': {
                    'x': int(box[0]),
                    'y': int(box[1]),
                    'width': int(box[2]),
                    'height': int(box[3])
                },
                'detection_confidence': float(detection_confidence),
                'predicted_name': predicted_name if is_verified else "unknown",
                'recognition_confidence': float(recognition_confidence),
                'verified': is_verified,
                'student_name': predicted_name if is_verified else None,  # Laravel compatibility
                'similarity': float(recognition_confidence),  # Laravel compatibility
                'all_predictions': all_predictions,
                'face_timing': {
                    'facenet_encoding_ms': round(facenet_time, 2),
                    'svm_classification_ms': round(svm_time, 2)
                }
            }
            
            results.append(face_result)
            
            print(f"  👤 Face {i+1}: {predicted_name} ({recognition_confidence:.3f}) - FaceNet: {facenet_time:.2f}ms, SVM: {svm_time:.2f}ms")
        
        # Calculate total processing time
        total_time = mtcnn_time + total_facenet_time + total_svm_time
        
        # Response dengan detailed timing
        response = {
            'success': True,
            'message': f'Processed {len(results)} face(s)',
            'results': results,
            'total_faces': len(results),
            'verified_faces': len([r for r in results if r['verified']]),
            'timing': {
                'mtcnn_detection_ms': round(mtcnn_time, 2),
                'facenet_encoding_ms': round(total_facenet_time, 2),
                'svm_classification_ms': round(total_svm_time, 2),
                'total_processing_ms': round(total_time, 2),
                'average_per_face_ms': round(total_time / len(results), 2) if results else 0
            },
            'performance_breakdown': {
                'mtcnn_percentage': round((mtcnn_time / total_time) * 100, 1) if total_time > 0 else 0,
                'facenet_percentage': round((total_facenet_time / total_time) * 100, 1) if total_time > 0 else 0,
                'svm_percentage': round((total_svm_time / total_time) * 100, 1) if total_time > 0 else 0
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return jsonify({
            'success': False,
            'message': f'Processing error: {str(e)}'
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get detailed model information"""
    try:
        info = model_loader.get_model_info()
        return jsonify({
            'success': True,
            'model_info': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎯 FACE RECOGNITION API - SIMPLE VERSION")
    print("📊 MTCNN + FaceNet + SVM (.pkl from main.ipynb)")
    print("="*60)
    
    if model_loader.is_loaded:
        model_info = model_loader.get_model_info()
        print(f"🏷️  Classes: {model_info['classes']}")
        print(f"🔢 Total classes: {model_info['num_classes']}")
        print(f"⚙️  SVM kernel: {model_info['svm_kernel']}")
    else:
        print("⚠️  Models not loaded - place .pkl files in models/ folder")
    
    print("\n📡 Available endpoints:")
    print("  - GET  /api/health")
    print("  - POST /api/verify-face")
    print("  - GET  /api/model-info")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)