# attendance-api-enhanced/utils/face_recognizer.py

import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from config import Config

class AdvancedFaceRecognizer:
    def __init__(self):
        self.svm_model = None
        self.label_encoder = None
        self.scaler = None
        self.threshold = Config.RECOGNITION_THRESHOLD
        self.model_path = 'models/advanced_face_recognizer.pkl'
        print("✓ Advanced SVM recognizer initialized")
        
    def train(self, encodings, labels, optimize_hyperparams=True):
        """Train advanced SVM with hyperparameter optimization"""
        try:
            print(f"Training SVM with {len(encodings)} samples...")
            
            if len(encodings) < 2 or len(set(labels)) < 2:
                print("✗ Need at least 2 different people for training")
                return False
                
            X = np.array(encodings)
            y = np.array(labels)
            
            print(f"Features shape: {X.shape}")
            print(f"Classes: {len(set(labels))} unique people")
            print(f"Samples per class: {[list(y).count(label) for label in set(labels)]}")
            
            # Check for invalid values
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                print("✗ Invalid values (NaN/inf) found in encodings")
                return False
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Feature scaling (important for SVM)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data if we have enough samples
            if len(X) >= 6:  # At least 6 samples for meaningful split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_encoded, 
                    test_size=0.2, 
                    random_state=42, 
                    stratify=y_encoded
                )
                use_validation = True
            else:
                X_train, y_train = X_scaled, y_encoded
                X_test, y_test = X_scaled, y_encoded
                use_validation = False
                print("Using all data for training (insufficient for split)")
            
            # Hyperparameter optimization
            if optimize_hyperparams and len(X_train) >= 4:
                print("Optimizing SVM hyperparameters...")
                self.svm_model = self.optimize_svm(X_train, y_train)
            else:
                print("Training SVM with default parameters...")
                self.svm_model = SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42,
                    C=1.0,
                    gamma='scale',
                    class_weight='balanced'  # Handle imbalanced classes
                )
                self.svm_model.fit(X_train, y_train)
            
            # Evaluate model
            if use_validation:
                y_pred = self.svm_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"✓ SVM training completed")
                print(f"Validation accuracy: {accuracy:.3f}")
                
                # Detailed classification report
                target_names = [self.label_encoder.inverse_transform([i])[0] for i in range(len(self.label_encoder.classes_))]
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
            else:
                print("✓ SVM training completed (no validation split)")
            
            # Save model
            self.save_model()
            
            return True
            
        except Exception as e:
            print(f"✗ SVM training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def optimize_svm(self, X_train, y_train):
        """Optimize SVM hyperparameters using GridSearch"""
        try:
            # Define parameter grid
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']
            }
            
            # Reduce grid for small datasets
            if len(X_train) < 20:
                param_grid = {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf']
                }
            
            # GridSearch with cross-validation
            svm = SVC(probability=True, random_state=42, class_weight='balanced')
            
            # Use smaller CV folds for small datasets
            cv_folds = min(3, len(X_train) // 2)
            cv_folds = max(2, cv_folds)  # At least 2 folds
            
            grid_search = GridSearchCV(
                svm, 
                param_grid, 
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            print(f"Hyperparameter optimization failed: {e}")
            # Fallback to default SVM
            svm = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                C=1.0,
                gamma='scale',
                class_weight='balanced'
            )
            svm.fit(X_train, y_train)
            return svm
    
    def predict(self, encoding, return_probabilities=False):
        """Predict person from encoding with confidence"""
        try:
            if self.svm_model is None or self.label_encoder is None or self.scaler is None:
                print("Model not trained")
                return None, 0.0
                
            # Prepare encoding
            encoding = np.array(encoding).reshape(1, -1)
            
            # Scale features
            encoding_scaled = self.scaler.transform(encoding)
            
            # Get prediction probabilities
            probabilities = self.svm_model.predict_proba(encoding_scaled)[0]
            
            # Get best prediction
            best_class_idx = np.argmax(probabilities)
            confidence = probabilities[best_class_idx]
            
            # Apply threshold
            if confidence < self.threshold:
                if return_probabilities:
                    return None, confidence, probabilities
                return None, confidence
            
            # Get label
            predicted_label = self.label_encoder.inverse_transform([best_class_idx])[0]
            
            if return_probabilities:
                # Return all class probabilities
                class_probs = {}
                for i, prob in enumerate(probabilities):
                    label = self.label_encoder.inverse_transform([i])[0]
                    class_probs[label] = float(prob)
                return predicted_label, confidence, class_probs
            
            return predicted_label, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            if return_probabilities:
                return None, 0.0, {}
            return None, 0.0
    
    def predict_batch(self, encodings):
        """Predict multiple encodings at once"""
        try:
            if self.svm_model is None or len(encodings) == 0:
                return []
            
            results = []
            for encoding in encodings:
                label, confidence = self.predict(encoding)
                results.append({
                    'label': label,
                    'confidence': float(confidence)
                })
            
            return results
            
        except Exception as e:
            print(f"Batch prediction error: {e}")
            return []
    
    def get_model_info(self):
        """Get information about the trained model"""
        if self.svm_model is None:
            return None
        
        info = {
            'classes': list(self.label_encoder.classes_),
            'n_classes': len(self.label_encoder.classes_),
            'n_support_vectors': self.svm_model.n_support_.tolist(),
            'total_support_vectors': int(np.sum(self.svm_model.n_support_)),
            'kernel': self.svm_model.kernel,
            'threshold': self.threshold
        }
        
        if hasattr(self.svm_model, 'C'):
            info['C'] = float(self.svm_model.C)
        if hasattr(self.svm_model, 'gamma'):
            info['gamma'] = self.svm_model.gamma
            
        return info
    
    def save_model(self):
        """Save trained model with all components"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'svm_model': self.svm_model,
                'label_encoder': self.label_encoder,
                'scaler': self.scaler,
                'threshold': self.threshold,
                'model_info': self.get_model_info()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            print(f"✓ Model saved to {self.model_path}")
            
        except Exception as e:
            print(f"✗ Save model error: {e}")
    
    def load_model(self):
        """Load trained model with all components"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Model file not found: {self.model_path}")
                return False
                
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.svm_model = model_data['svm_model']
            self.label_encoder = model_data['label_encoder']
            self.scaler = model_data['scaler']
            self.threshold = model_data.get('threshold', Config.RECOGNITION_THRESHOLD)
            
            print("✓ Advanced SVM model loaded successfully")
            
            # Print model info
            info = model_data.get('model_info', {})
            if info:
                print(f"Classes: {info.get('n_classes', 0)}")
                print(f"Support vectors: {info.get('total_support_vectors', 0)}")
                print(f"Kernel: {info.get('kernel', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"✗ Load model error: {e}")
            return False
    
    def update_threshold(self, new_threshold):
        """Update recognition threshold"""
        self.threshold = new_threshold
        print(f"Recognition threshold updated to: {new_threshold}")
        
    def validate_model(self):
        """Validate that model is properly loaded"""
        try:
            if self.svm_model is None:
                return False
            
            # Test with dummy data
            dummy_encoding = np.random.rand(512)  # FaceNet output size
            label, confidence = self.predict(dummy_encoding)
            
            print("✓ Model validation successful")
            return True
            
        except Exception as e:
            print(f"✗ Model validation failed: {e}")
            return False