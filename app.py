# app.py - SIMPLE VERSION (MTCNN + FaceNet + SVM)
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
from datetime import datetime

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
    Main endpoint untuk face recognition
    Input: base64 image
    Output: detected faces dengan identity predictions
    """
    try:
        # Validate input
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'message': 'No image provided'
            }), 400
        
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
        
        # Detect faces
        faces = face_detector.detect_faces(image)
        
        if not faces:
            return jsonify({
                'success': True,
                'message': 'No faces detected',
                'results': []
            })
        
        print(f"👥 Detected {len(faces)} face(s)")
        
        # Process each face
        results = []
        for i, face_data in enumerate(faces):
            face_crop = face_data['face']
            box = face_data['box']
            detection_confidence = face_data['confidence']
            
            # Preprocess face for FaceNet
            processed_face = face_detector.preprocess_face(face_crop)
            if processed_face is None:
                continue
            
            # Generate embedding
            embedding = face_encoder.encode_face(processed_face)
            if embedding is None:
                continue
            
            # Predict identity dengan threshold rendah untuk SVM
            predicted_name, recognition_confidence = model_loader.predict(
                embedding, threshold=0.15  # Threshold rendah untuk SVM
            )
            
            # Get all predictions untuk analisis
            all_predictions = model_loader.get_all_predictions(embedding)
            
            # Enhanced verification logic
            is_verified = False
            if predicted_name != "unknown":
                # Cek confidence gap - apakah prediksi terbaik cukup berbeda dari yang kedua
                sorted_preds = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_preds) >= 2:
                    confidence_gap = sorted_preds[0][1] - sorted_preds[1][1]
                    # Verifikasi jika confidence >= 0.08 DAN gap >= 0.03
                    is_verified = recognition_confidence >= 0.08 and confidence_gap >= 0.03
                else:
                    # Hanya 1 class, threshold lebih rendah
                    is_verified = recognition_confidence >= 0.10
                    
                print(f"  🔍 Verification analysis:")
                print(f"      Confidence: {recognition_confidence:.3f}")
                if len(sorted_preds) >= 2:
                    print(f"      Gap: {confidence_gap:.3f}")
                print(f"      Verified: {is_verified}")
            
            # Hasil untuk face ini
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
                'student_name': predicted_name if is_verified else None,  # Untuk Laravel compatibility
                'similarity': float(recognition_confidence),  # Untuk Laravel compatibility
                'all_predictions': all_predictions
            }
            
            results.append(face_result)
            
            print(f"  👤 Face {i+1}: {predicted_name} ({recognition_confidence:.3f})")
        
        # Response
        response = {
            'success': True,
            'message': f'Processed {len(results)} face(s)',
            'results': results,
            'total_faces': len(results),
            'verified_faces': len([r for r in results if r['verified']])
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