# utils/model_loader.py
import pickle
import json
import os
import numpy as np
from datetime import datetime

class ModelLoader:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.svm_model = None
        self.label_encoder = None
        self.metadata = None
        self.is_loaded = False
        
        # Paths
        self.svm_path = os.path.join(model_dir, "svm_face_model.pkl")
        self.encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        self.metadata_path = os.path.join(model_dir, "model_metadata.json")
        
        # Buat folder jika belum ada
        os.makedirs(model_dir, exist_ok=True)
        
    def load_models(self):
        """Load semua model components"""
        try:
            print("🔄 Loading trained models...")
            
            # Check files exist
            if not os.path.exists(self.svm_path):
                print(f"❌ SVM model not found: {self.svm_path}")
                return False
                
            if not os.path.exists(self.encoder_path):
                print(f"❌ Label encoder not found: {self.encoder_path}")
                return False
            
            # Load SVM model
            print("  → Loading SVM model...")
            with open(self.svm_path, 'rb') as f:
                self.svm_model = pickle.load(f)
            
            # Load label encoder
            print("  → Loading label encoder...")
            with open(self.encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load metadata (optional)
            if os.path.exists(self.metadata_path):
                print("  → Loading metadata...")
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            self.is_loaded = True
            
            # Print info
            print("✅ Models loaded successfully!")
            print(f"  🏷️ Classes: {list(self.label_encoder.classes_)}")
            print(f"  🎯 SVM kernel: {self.svm_model.kernel}")
            if self.metadata:
                print(f"  📊 Test accuracy: {self.metadata.get('test_accuracy', 'N/A')}")
                print(f"  📅 Training date: {self.metadata.get('training_date', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, face_encoding, threshold=0.15):
        """
        Predict identity dari face encoding dengan adaptive threshold
        """
        if not self.is_loaded:
            print("❌ Models not loaded!")
            return None, 0.0
        
        try:
            # Reshape encoding untuk SVM
            encoding = np.array(face_encoding).reshape(1, -1)
            
            # Get prediction probabilities
            probabilities = self.svm_model.predict_proba(encoding)[0]
            
            # Get best prediction
            best_idx = np.argmax(probabilities)
            confidence = probabilities[best_idx]
            predicted_label = self.label_encoder.inverse_transform([best_idx])[0]
            
            # Get second best untuk confidence gap analysis
            sorted_probs = np.sort(probabilities)[::-1]
            confidence_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else confidence
            
            print(f"🔍 Prediction Debug:")
            print(f"   Best: {predicted_label} ({confidence:.3f})")
            print(f"   Gap: {confidence_gap:.3f}")
            print(f"   Threshold: {threshold}")
            
            # Adaptive thresholding berdasarkan confidence gap
            if confidence >= threshold and confidence_gap >= 0.02:
                return predicted_label, confidence
            else:
                # Jika confidence rendah tapi masih ada gap yang cukup
                if confidence >= 0.08 and confidence_gap >= 0.03:
                    return predicted_label, confidence
                else:
                    return "unknown", confidence
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, 0.0
    
    def get_all_predictions(self, face_encoding):
        """Get all class predictions dengan confidence"""
        if not self.is_loaded:
            return {}
        
        try:
            encoding = np.array(face_encoding).reshape(1, -1)
            probabilities = self.svm_model.predict_proba(encoding)[0]
            
            results = {}
            for i, prob in enumerate(probabilities):
                label = self.label_encoder.inverse_transform([i])[0]
                results[label] = float(prob)
            
            return results
            
        except Exception as e:
            print(f"❌ Error getting all predictions: {e}")
            return {}
    
    def get_model_info(self):
        """Get informasi tentang model"""
        if not self.is_loaded:
            return None
        
        info = {
            "model_loaded": True,
            "classes": list(self.label_encoder.classes_),
            "num_classes": len(self.label_encoder.classes_),
            "svm_kernel": self.svm_model.kernel,
            "model_type": "SVM from main.ipynb"
        }
        
        if self.metadata:
            info.update({
                "training_date": self.metadata.get("training_date"),
                "test_accuracy": self.metadata.get("test_accuracy"),
                "cv_score": self.metadata.get("cv_score")
            })
        
        return info