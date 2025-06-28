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
from utils.scalable_face_recognizer import ScalableFaceRecognizer
from services.migration_service import MigrationService

app = Flask(__name__)
CORS(app)

# Initialize services
print("Initializing Face Recognition API...")
print("Using: MTCNN + FaceNet + Advanced SVM")

face_detector = FaceDetector()
face_encoder = FaceNetEncoder()
face_recognizer = AdvancedFaceRecognizer()
dataset_processor = EnhancedDatasetProcessor()
scalable_recognizer = ScalableFaceRecognizer()
migration_service = MigrationService()

print("\n🔄 Loading verification database...")
scalable_recognizer.load_database()

# Auto-load model jika ada
model_path = 'models/advanced_face_recognizer.pkl'
if os.path.exists(model_path):
    print(f"\n🔄 Loading existing model...")
    load_success = face_recognizer.load_model()
    if load_success:
        # Update threshold dari config jika berbeda
        if face_recognizer.threshold != Config.RECOGNITION_THRESHOLD:
            print(f"🎯 Updating threshold: {face_recognizer.threshold} → {Config.RECOGNITION_THRESHOLD}")
            face_recognizer.update_threshold(Config.RECOGNITION_THRESHOLD)
        print(f"✅ Model loaded successfully")
    else:
        print(f"❌ Model loading failed")
else:
    print(f"📝 No existing model found - train model first")

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
        'config': {
            'recognition_threshold': Config.RECOGNITION_THRESHOLD,
            'face_confidence_threshold': Config.FACE_CONFIDENCE_THRESHOLD,
            'dataset_path': Config.DATASET_PATH
        }
    })

@app.route('/api/validate-dataset', methods=['GET'])
def validate_dataset():
    """Validate dataset structure"""
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

@app.route('/api/analyze-dataset', methods=['GET'])
def analyze_dataset_quality():
    """Analisis kualitas dataset"""
    try:
        print("🔍 Starting dataset quality analysis...")
        is_valid, quality_report = dataset_processor.validate_and_analyze_dataset()
        
        if quality_report is None:
            return jsonify({
                'success': False,
                'message': 'Dataset tidak valid atau tidak ditemukan'
            }), 400
        
        # Generate recommendations
        avg_quality = quality_report['summary']['avg_quality_score']
        if avg_quality < 50:
            recommendations = [
                "Kualitas dataset buruk - perlu perbaikan menyeluruh",
                "Gunakan pencahayaan yang baik dan konsisten",
                "Pastikan wajah mengisi minimal 10% area gambar"
            ]
        elif avg_quality < 70:
            recommendations = [
                "Kualitas dataset cukup - perlu beberapa perbaikan",
                "Tingkatkan kualitas gambar dengan skor rendah"
            ]
        else:
            recommendations = ["Kualitas dataset baik - siap untuk training!"]
        
        return jsonify({
            'success': True,
            'message': 'Analisis dataset selesai',
            'dataset_valid': is_valid,
            'quality_report': quality_report,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during analysis: {str(e)}'
        }), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train model from dataset"""
    try:
        data = request.get_json() if request.is_json else {}
        optimize_hyperparams = data.get('optimize_hyperparams', True)
        
        print("🚀 Starting training process...")
        
        # Validate dataset
        if not dataset_processor.validate_dataset():
            return jsonify({
                'success': False,
                'message': 'Dataset validation failed'
            }), 400
        
        # Process dataset
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
        
        # Train model
        success = face_recognizer.train(encodings, labels, optimize_hyperparams)
        
        if success:
            # Update threshold sesuai config
            face_recognizer.update_threshold(Config.RECOGNITION_THRESHOLD)
            face_recognizer.save_model()  # Save ulang dengan threshold yang benar
            
            return jsonify({
                'success': True,
                'message': f'Model trained successfully with {len(encodings)} face encodings',
                'statistics': {
                    'total_encodings': len(encodings),
                    'unique_students': len(set(labels)),
                    'students': list(set(labels)),
                    'threshold_used': Config.RECOGNITION_THRESHOLD
                },
                'model_info': face_recognizer.get_model_info()
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

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Recognize face from image"""
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
        
        # Detect faces
        faces = face_detector.detect_faces(image)
        if not faces:
            return jsonify({
                'success': False,
                'message': 'No faces detected'
            })
        
        results = []
        for i, face_detection in enumerate(faces):
            # Extract and encode face
            face = face_detector.extract_face(image, face_detection)
            if face is None:
                continue
                
            encoding = face_encoder.encode_face(face)
            if encoding is None:
                continue
            
            # Recognize
            student_name, confidence, probabilities = face_recognizer.predict(
                encoding, return_probabilities=True
            )
            
            result = {
                'face_id': i,
                'student_name': student_name,
                'confidence': float(confidence),
                'above_threshold': confidence >= face_recognizer.threshold,
                'bounding_box': face_detection['box'],
                'mtcnn_confidence': float(face_detection['confidence'])
            }
            
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
            'threshold_used': face_recognizer.threshold,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Recognition error: {e}")
        return jsonify({
            'success': False,
            'message': f'Recognition error: {str(e)}'
        }), 500

@app.route('/api/benchmark', methods=['POST'])
def benchmark_model():
    """Benchmark model performance"""
    try:
        print("🏃‍♂️ STARTING BENCHMARK")
        
        if face_recognizer.svm_model is None:
            return jsonify({
                'success': False,
                'message': 'Model belum di-train'
            }), 400
        
        # Load dataset
        encodings, labels = dataset_processor.process_dataset()
        if len(encodings) == 0:
            return jsonify({
                'success': False,
                'message': 'No data for benchmarking'
            })
        
        print(f"📊 Testing {len(encodings)} samples with threshold: {face_recognizer.threshold}")
        
        # Test predictions
        correct_all = 0
        correct_threshold = 0
        above_threshold = 0
        confidences = []
        
        for encoding, true_label in zip(encodings, labels):
            try:
                predicted_label, confidence = face_recognizer.predict(encoding)
                
                if predicted_label == true_label:
                    correct_all += 1
                
                confidences.append(confidence)
                
                if confidence >= face_recognizer.threshold:
                    above_threshold += 1
                    if predicted_label == true_label:
                        correct_threshold += 1
                        
            except Exception as e:
                print(f"Prediction error: {e}")
                continue
        
        # Calculate metrics
        total_samples = len(encodings)
        accuracy_all = correct_all / total_samples
        accuracy_threshold = correct_threshold / above_threshold if above_threshold > 0 else 0
        coverage = above_threshold / total_samples
        
        confidence_stats = {
            'min': float(min(confidences)) if confidences else 0,
            'max': float(max(confidences)) if confidences else 0,
            'avg': float(np.mean(confidences)) if confidences else 0,
            'threshold': face_recognizer.threshold
        }
        
        results = {
            'accuracy_all': float(accuracy_all),
            'accuracy_threshold': float(accuracy_threshold),
            'coverage': float(coverage),
            'total_samples': total_samples,
            'above_threshold_samples': above_threshold,
            'confidence_stats': confidence_stats
        }
        
        print(f"📊 Results: All={accuracy_all:.1%}, Threshold={accuracy_threshold:.1%}, Coverage={coverage:.1%}")
        
        return jsonify({
            'success': True,
            'message': f'Benchmark completed - Overall: {accuracy_all:.1%}, Coverage: {coverage:.1%}',
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Benchmark error: {e}")
        return jsonify({
            'success': False,
            'message': f'Benchmark error: {str(e)}'
        }), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    try:
        model_info = face_recognizer.get_model_info()
        
        if model_info is None:
            return jsonify({
                'success': False,
                'message': 'Model belum di-train'
            })
        
        # Test encoder stability
        dummy_face = np.random.randint(0, 255, (*face_encoder.image_size, 3), dtype=np.uint8)
        test_encodings = []
        
        for i in range(3):
            encoding = face_encoder.encode_face(dummy_face)
            if encoding is not None:
                test_encodings.append(encoding)
        
        stability_score = 0.0
        if len(test_encodings) >= 2:
            similarities = []
            for i in range(len(test_encodings)):
                for j in range(i+1, len(test_encodings)):
                    similarity = np.dot(test_encodings[i], test_encodings[j])
                    similarities.append(similarity)
            stability_score = float(np.mean(similarities))
        
        return jsonify({
            'success': True,
            'model_info': model_info,
            'encoder_info': {
                'model_type': getattr(face_encoder, 'model_type', 'unknown'),
                'image_size': face_encoder.image_size,
                'encoding_dimension': 512,
                'stability_score': stability_score
            },
            'config_info': {
                'threshold_from_config': Config.RECOGNITION_THRESHOLD,
                'threshold_current': face_recognizer.threshold
            },
            'training_status': 'READY',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting model info: {str(e)}'
        }), 500

@app.route('/api/set-threshold', methods=['POST'])
def set_threshold():
    """Set recognition threshold"""
    try:
        data = request.get_json()
        new_threshold = data.get('threshold', Config.RECOGNITION_THRESHOLD)
        
        if not 0.1 <= new_threshold <= 0.9:
            return jsonify({
                'success': False,
                'message': 'Threshold harus antara 0.1 dan 0.9'
            }), 400
        
        old_threshold = face_recognizer.threshold
        face_recognizer.update_threshold(new_threshold)
        
        print(f"🎯 Threshold updated: {old_threshold} → {new_threshold}")
        
        return jsonify({
            'success': True,
            'message': f'Threshold updated from {old_threshold} to {new_threshold}',
            'old_threshold': old_threshold,
            'new_threshold': new_threshold
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/migrate-to-verification', methods=['POST'])
def migrate_to_verification():
    """Migrate from classification to verification approach"""
    try:
        print("🔄 Starting migration to verification...")
        
        result = migration_service.migrate_to_verification()
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': result['message'],
                'statistics': result['statistics'],
                'migration_details': result['migration_details'],
                'next_steps': [
                    'Use /api/verify-face for face recognition',
                    'Use /api/add-student to add new students',
                    'Use /api/verification-stats for database info'
                ]
            })
        else:
            return jsonify({
                'success': False,
                'message': result['message']
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Migration error: {str(e)}'
        }), 500

@app.route('/api/verify-face', methods=['POST'])
def verify_face():
    """Verify face using verification approach - Fixed JSON serialization"""
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
        
        print(f"🔍 Processing image for verification: {image.shape}")
        
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
                
                # Verify using scalable recognizer
                student_id, student_name, similarity, top_matches = scalable_recognizer.verify_face(encoding, top_k=5)
                
                # Convert all values to JSON-safe types
                result = {
                    'face_id': int(i),
                    'student_id': str(student_id) if student_id else None,
                    'student_name': str(student_name) if student_name else None,
                    'similarity': float(similarity) if similarity is not None else 0.0,
                    'verified': bool(similarity >= scalable_recognizer.verification_threshold) if similarity is not None else False,
                    'bounding_box': [int(x) for x in face_detection['box']],  # Convert to regular int
                    'mtcnn_confidence': float(face_detection['confidence']),
                    'verification_threshold': float(scalable_recognizer.verification_threshold)
                }
                
                # Add top matches with JSON-safe conversion
                if top_matches:
                    result['top_matches'] = []
                    for match in top_matches[:3]:
                        safe_match = {
                            'student_name': str(match['student_name']),
                            'similarity': float(match['max_similarity']),
                            'encoding_count': int(match['encoding_count'])
                        }
                        result['top_matches'].append(safe_match)
                
                results.append(result)
                
            except Exception as face_error:
                print(f"Error processing face {i}: {face_error}")
                continue
        
        # Get database stats with JSON-safe conversion
        try:
            raw_stats = scalable_recognizer.get_statistics()
            safe_stats = {
                'total_students': int(raw_stats['total_students']),
                'total_encodings': int(raw_stats['total_encodings']),
                'avg_encodings_per_student': float(raw_stats['avg_encodings_per_student']),
                'verification_threshold': float(raw_stats['verification_threshold'])
            }
        except:
            safe_stats = {}
        
        return jsonify({
            'success': True,
            'message': f'Verification completed for {len(results)} face(s)',
            'results': results,
            'database_stats': safe_stats,
            'timestamp': datetime.now().isoformat()
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
        
        required_fields = ['student_name', 'images']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'message': f'Missing required field: {field}'
                }), 400
        
        student_name = data['student_name']
        images = data['images']  # List of base64 images
        nim = data.get('nim', f'NIM_{student_name.upper()}')
        kelas = data.get('class', 'Default_Class')
        
        # Process images to get encodings
        encodings = []
        processed_count = 0
        
        for i, image_data in enumerate(images):
            try:
                # Decode image
                image = decode_base64_image(image_data)
                if image is None:
                    continue
                
                # Detect and extract face
                faces = face_detector.detect_faces(image)
                if not faces:
                    continue
                
                # Use best face
                best_face = max(faces, key=lambda x: x['confidence'])
                face = face_detector.extract_face(image, best_face)
                if face is None:
                    continue
                
                # Generate encoding
                encoding = face_encoder.encode_face(face)
                if encoding is not None:
                    encodings.append(encoding)
                    processed_count += 1
                    
            except Exception as img_error:
                print(f"Error processing image {i}: {img_error}")
                continue
        
        if len(encodings) == 0:
            return jsonify({
                'success': False,
                'message': 'No valid face encodings could be generated from provided images'
            }), 400
        
        # Add to database
        student_id = f"student_{student_name.lower().replace(' ', '_')}"
        success = scalable_recognizer.add_student(
            student_id=student_id,
            student_name=student_name,
            nim=nim,
            kelas=kelas,
            encodings=encodings
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Student {student_name} added successfully',
                'student_info': {
                    'student_id': student_id,
                    'student_name': student_name,
                    'nim': nim,
                    'class': kelas,
                    'encoding_count': len(encodings),
                    'processed_images': processed_count,
                    'total_images': len(images)
                },
                'database_stats': scalable_recognizer.get_statistics()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to add student to database'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Add student error: {str(e)}'
        }), 500

@app.route('/api/verification-stats', methods=['GET'])
def get_verification_stats():
    """Get verification database statistics - Fixed JSON serialization"""
    try:
        raw_stats = scalable_recognizer.get_statistics()
        
        # Convert to JSON-safe types
        safe_stats = {
            'total_students': int(raw_stats['total_students']),
            'total_encodings': int(raw_stats['total_encodings']),
            'avg_encodings_per_student': float(raw_stats['avg_encodings_per_student']),
            'verification_threshold': float(raw_stats['verification_threshold']),
            'database_size_mb': float(raw_stats.get('database_size_mb', 0.0))
        }
        
        # Convert encoding distribution
        safe_distribution = {}
        for name, count in raw_stats.get('encoding_distribution', {}).items():
            safe_distribution[str(name)] = int(count)
        safe_stats['encoding_distribution'] = safe_distribution
        
        return jsonify({
            'success': True,
            'message': 'Verification statistics retrieved',
            'statistics': safe_stats,
            'threshold_info': {
                'current_threshold': float(scalable_recognizer.verification_threshold),
                'threshold_type': 'cosine_similarity',
                'range': '0.0 to 1.0 (higher is better)',
                'recommended': '0.70 - 0.80'
            },
            'scalability_info': {
                'current_students': int(raw_stats['total_students']),
                'max_recommended': 'No limit (scales linearly)',
                'performance': f'O({int(raw_stats["total_students"])}) per verification'
            }
        })
        
    except Exception as e:
        print(f"Stats error: {e}")
        return jsonify({
            'success': False,
            'message': f'Stats error: {str(e)}'
        }), 500

@app.route('/api/test-verification', methods=['POST'])
def test_verification():
    """Test verification system with current dataset - Fixed JSON serialization"""
    try:
        result = migration_service.test_verification_system()
        
        if result['success']:
            # Convert benchmark results to JSON-safe types
            benchmark = result['benchmark_results']
            safe_benchmark = {
                'accuracy': float(benchmark['accuracy']),
                'precision': float(benchmark['precision']),
                'recall': float(benchmark['recall']),
                'f1_score': float(benchmark['f1_score']),
                'total_tests': int(benchmark['total_tests']),
                'correct_verifications': int(benchmark['correct_verifications']),
                'false_positives': int(benchmark['false_positives']),
                'false_negatives': int(benchmark['false_negatives']),
                'avg_similarity': float(benchmark['avg_similarity'])
            }
            
            # Convert similarity stats
            if 'similarity_stats' in benchmark:
                safe_benchmark['similarity_stats'] = {
                    'min': float(benchmark['similarity_stats']['min']),
                    'max': float(benchmark['similarity_stats']['max']),
                    'std': float(benchmark['similarity_stats']['std'])
                }
            
            # Convert detailed results
            if 'detailed_results' in benchmark:
                safe_detailed = []
                for detail in benchmark['detailed_results'][:5]:  # First 5 only
                    safe_detail = {
                        'index': int(detail['index']),
                        'true_label': str(detail['true_label']),
                        'predicted_name': str(detail['predicted_name']) if detail['predicted_name'] else None,
                        'similarity': float(detail['similarity']),
                        'verified': bool(detail['verified']),
                        'correct': bool(detail['correct'])
                    }
                    safe_detailed.append(safe_detail)
                safe_benchmark['detailed_results'] = safe_detailed
            
            return jsonify({
                'success': True,
                'message': result['message'],
                'benchmark_results': safe_benchmark,
                'recommendations': [
                    f"Accuracy: {safe_benchmark['accuracy']:.1%}",
                    f"Precision: {safe_benchmark['precision']:.1%}",
                    f"Recall: {safe_benchmark['recall']:.1%}",
                    "System ready for production!" if safe_benchmark['accuracy'] > 0.9 else "Consider adding more training images"
                ]
            })
        else:
            return jsonify({
                'success': False,
                'message': result['message']
            }), 400
            
    except Exception as e:
        print(f"Test verification error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Test error: {str(e)}'
        }), 500

@app.route('/api/remove-student', methods=['POST'])
def remove_student():
    """Remove student from verification database"""
    try:
        data = request.get_json()
        
        if 'student_id' not in data:
            return jsonify({
                'success': False,
                'message': 'Missing student_id'
            }), 400
        
        student_id = data['student_id']
        success = scalable_recognizer.remove_student(student_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Student {student_id} removed successfully',
                'database_stats': scalable_recognizer.get_statistics()
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Student {student_id} not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Remove student error: {str(e)}'
        }), 500

@app.route('/api/set-verification-threshold', methods=['POST'])
def set_verification_threshold():
    """Set verification threshold"""
    try:
        data = request.get_json()
        new_threshold = data.get('threshold', 0.75)
        
        if not 0.1 <= new_threshold <= 1.0:
            return jsonify({
                'success': False,
                'message': 'Threshold must be between 0.1 and 1.0'
            }), 400
        
        old_threshold = scalable_recognizer.verification_threshold
        scalable_recognizer.update_threshold(new_threshold)
        
        return jsonify({
            'success': True,
            'message': f'Verification threshold updated from {old_threshold} to {new_threshold}',
            'old_threshold': old_threshold,
            'new_threshold': new_threshold,
            'threshold_info': {
                'type': 'cosine_similarity',
                'range': '0.0 to 1.0',
                'current': new_threshold
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
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
    print("- GET  /api/analyze-dataset")
    print("- POST /api/train")
    print("- POST /api/recognize")
    print("- POST /api/benchmark")
    print("- GET  /api/model-info")
    print("- POST /api/set-threshold")
    print("="*60)
    
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)