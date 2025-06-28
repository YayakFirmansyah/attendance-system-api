# utils/scalable_face_recognizer.py
import numpy as np
import pickle
import os
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

class ScalableFaceRecognizer:
    def __init__(self):
        self.student_encodings = {}  # {student_id: [list_of_encodings]}
        self.student_metadata = {}   # {student_id: {name, nim, class, etc}}
        self.verification_threshold = Config.RECOGNITION_THRESHOLD  # Cosine similarity threshold
        self.database_path = 'models/face_database.pkl'
        self.metadata_path = 'models/student_metadata.json'
        print("✓ Scalable Face Recognizer initialized (Verification-based)")
    
    def make_json_safe(self, obj):
        """Convert numpy types to JSON-safe Python types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self.make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.make_json_safe(item) for item in obj]
        else:
            return obj

    def add_student(self, student_id, student_name, nim=None, kelas=None, encodings=None):
        """Add new student to database"""
        try:
            if encodings is None or len(encodings) == 0:
                print(f"❌ No encodings provided for {student_name}")
                return False
            
            # Convert encodings to numpy arrays
            processed_encodings = []
            for encoding in encodings:
                if isinstance(encoding, np.ndarray):
                    processed_encodings.append(encoding)
                else:
                    processed_encodings.append(np.array(encoding))
            
            # Store encodings
            self.student_encodings[student_id] = processed_encodings
            
            # Store metadata
            self.student_metadata[student_id] = {
                'name': student_name,
                'nim': nim,
                'class': kelas,
                'added_date': datetime.now().isoformat(),
                'encoding_count': len(processed_encodings)
            }
            
            print(f"✅ Added {student_name} with {len(processed_encodings)} encodings")
            self.save_database()
            return True
            
        except Exception as e:
            print(f"❌ Error adding student: {e}")
            return False
    
    def remove_student(self, student_id):
        """Remove student from database"""
        try:
            if student_id in self.student_encodings:
                student_name = self.student_metadata.get(student_id, {}).get('name', 'Unknown')
                del self.student_encodings[student_id]
                if student_id in self.student_metadata:
                    del self.student_metadata[student_id]
                self.save_database()
                print(f"✅ Removed {student_name} from database")
                return True
            else:
                print(f"❌ Student {student_id} not found")
                return False
        except Exception as e:
            print(f"❌ Error removing student: {e}")
            return False
    
    def verify_face(self, input_encoding, top_k=5):
        """Verify face against all students in database - JSON safe version"""
        try:
            if len(self.student_encodings) == 0:
                print("⚠️ No students in database")
                return None, None, 0.0, []

            # Ensure input encoding is numpy array and normalized
            if not isinstance(input_encoding, np.ndarray):
                input_encoding = np.array(input_encoding)

            input_norm = np.linalg.norm(input_encoding)
            if input_norm > 0:
                input_encoding = input_encoding / input_norm

            results = []

            for student_id, stored_encodings in self.student_encodings.items():
                student_metadata = self.student_metadata.get(student_id, {})
                student_name = student_metadata.get('name', 'Unknown')

                similarities = []
                for stored_encoding in stored_encodings:
                    stored_norm = np.linalg.norm(stored_encoding)
                    if stored_norm > 0:
                        normalized_stored = stored_encoding / stored_norm
                    else:
                        continue
                    similarity = np.dot(input_encoding, normalized_stored)
                    similarities.append(float(similarity))

                if not similarities:
                    continue

                max_similarity = max(similarities)
                avg_similarity = sum(similarities) / len(similarities)

                results.append({
                    'student_id': str(student_id),
                    'student_name': str(student_name),
                    'max_similarity': float(max_similarity),
                    'avg_similarity': float(avg_similarity),
                    'encoding_count': int(len(similarities)),
                    'metadata': self.make_json_safe(student_metadata)
                })

            results.sort(key=lambda x: x['max_similarity'], reverse=True)

            if results:
                best_match = results[0]
                best_similarity = float(best_match['max_similarity'])

                print(f"🔍 Best match: {best_match['student_name']} (similarity: {best_similarity:.3f})")

                if best_similarity >= self.verification_threshold:
                    return (
                        best_match['student_id'],
                        best_match['student_name'],
                        best_similarity,
                        results[:top_k]
                    )
                else:
                    print(f"⚠️ Below threshold: {best_similarity:.3f} < {self.verification_threshold}")
                    return None, None, best_similarity, results[:top_k]

            return None, None, 0.0, []

        except Exception as e:
            print(f"❌ Verification error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0.0, []
    
    def batch_verify(self, encodings):
        """Verify multiple faces at once"""
        results = []
        for i, encoding in enumerate(encodings):
            student_id, student_name, similarity, top_matches = self.verify_face(encoding)
            results.append({
                'index': i,
                'student_id': student_id,
                'student_name': student_name,
                'similarity': float(similarity) if similarity else 0.0,
                'verified': similarity >= self.verification_threshold if similarity else False
            })
        return results
    
    def get_statistics(self):
        """Get database statistics"""
        total_students = len(self.student_encodings)
        total_encodings = sum(len(encodings) for encodings in self.student_encodings.values())
        
        encoding_distribution = {}
        for student_id, encodings in self.student_encodings.items():
            student_metadata = self.student_metadata.get(student_id, {})
            student_name = student_metadata.get('name', student_id)
            encoding_distribution[student_name] = len(encodings)
        
        return {
            'total_students': total_students,
            'total_encodings': total_encodings,
            'avg_encodings_per_student': total_encodings / total_students if total_students > 0 else 0,
            'verification_threshold': self.verification_threshold,
            'encoding_distribution': encoding_distribution,
            'database_size_mb': self.get_database_size()
        }
    
    def get_database_size(self):
        """Calculate database size in MB"""
        try:
            total_size = 0
            if os.path.exists(self.database_path):
                total_size += os.path.getsize(self.database_path)
            if os.path.exists(self.metadata_path):
                total_size += os.path.getsize(self.metadata_path)
            return round(total_size / (1024 * 1024), 2)
        except:
            return 0.0
    
    def benchmark_verification(self, test_encodings, test_labels):
        """Benchmark verification accuracy"""
        if len(test_encodings) != len(test_labels):
            print("❌ Mismatch between encodings and labels count")
            return {}
        
        correct_verifications = 0
        total_verifications = 0
        false_positives = 0
        false_negatives = 0
        
        results = []
        similarity_scores = []
        
        # Get all known student names
        known_students = set()
        for metadata in self.student_metadata.values():
            known_students.add(metadata.get('name', ''))
        
        print(f"🧪 Testing {len(test_encodings)} samples against {len(known_students)} known students")
        
        for i, (encoding, true_label) in enumerate(zip(test_encodings, test_labels)):
            try:
                student_id, student_name, similarity, top_matches = self.verify_face(encoding)
                
                is_verified = similarity >= self.verification_threshold if similarity else False
                is_correct = student_name == true_label if student_name else False
                is_known_student = true_label in known_students
                
                total_verifications += 1
                similarity_scores.append(similarity if similarity else 0.0)
                
                if is_verified and is_correct:
                    correct_verifications += 1
                elif is_verified and not is_correct:
                    false_positives += 1
                elif not is_verified and is_known_student:
                    false_negatives += 1
                
                results.append({
                    'index': i,
                    'true_label': true_label,
                    'predicted_name': student_name,
                    'similarity': float(similarity) if similarity else 0.0,
                    'verified': is_verified,
                    'correct': is_correct,
                    'known_student': is_known_student
                })
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(test_encodings)} samples")
                    
            except Exception as e:
                print(f"❌ Error testing sample {i}: {e}")
                continue
        
        # Calculate metrics
        accuracy = correct_verifications / total_verifications if total_verifications > 0 else 0
        precision = correct_verifications / (correct_verifications + false_positives) if (correct_verifications + false_positives) > 0 else 0
        recall = correct_verifications / (correct_verifications + false_negatives) if (correct_verifications + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'total_tests': total_verifications,
            'correct_verifications': correct_verifications,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'avg_similarity': float(np.mean(similarity_scores)) if similarity_scores else 0.0,
            'similarity_stats': {
                'min': float(min(similarity_scores)) if similarity_scores else 0.0,
                'max': float(max(similarity_scores)) if similarity_scores else 0.0,
                'std': float(np.std(similarity_scores)) if similarity_scores else 0.0
            },
            'detailed_results': results[:10]  # First 10 for brevity
        }
    
    def update_threshold(self, new_threshold):
        """Update verification threshold"""
        if not 0.0 <= new_threshold <= 1.0:
            print(f"❌ Invalid threshold: {new_threshold}. Must be between 0.0 and 1.0")
            return False
        
        old_threshold = self.verification_threshold
        self.verification_threshold = new_threshold
        print(f"🎯 Verification threshold updated: {old_threshold} → {new_threshold}")
        return True
    
    def save_database(self):
        """Save student database"""
        try:
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            
            # Save encodings
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.student_encodings, f)
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.student_metadata, f, indent=2)
            
            print(f"✅ Database saved ({len(self.student_encodings)} students)")
            
        except Exception as e:
            print(f"❌ Save database error: {e}")
    
    def load_database(self):
        """Load student database"""
        try:
            # Load encodings
            if os.path.exists(self.database_path):
                with open(self.database_path, 'rb') as f:
                    self.student_encodings = pickle.load(f)
                print(f"✅ Encodings loaded from {self.database_path}")
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.student_metadata = json.load(f)
                print(f"✅ Metadata loaded from {self.metadata_path}")
            
            total_students = len(self.student_encodings)
            if total_students > 0:
                print(f"📊 Database loaded: {total_students} students")
                
                # Print summary
                for student_id, metadata in self.student_metadata.items():
                    student_name = metadata.get('name', student_id)
                    encoding_count = len(self.student_encodings.get(student_id, []))
                    print(f"   - {student_name}: {encoding_count} encodings")
                
                return True
            else:
                print("📝 No existing database found")
                return False
                
        except Exception as e:
            print(f"❌ Load database error: {e}")
            return False
    
    def get_student_list(self):
        """Get list of all students in database"""
        students = []
        for student_id, metadata in self.student_metadata.items():
            students.append({
                'student_id': student_id,
                'student_name': metadata.get('name', 'Unknown'),
                'nim': metadata.get('nim', ''),
                'class': metadata.get('class', ''),
                'encoding_count': len(self.student_encodings.get(student_id, [])),
                'added_date': metadata.get('added_date', '')
            })
        
        # Sort by name
        students.sort(key=lambda x: x['student_name'])
        return students
    
    def migrate_from_classification(self, classification_model, dataset_processor):
        """Migrate data from classification model to verification database"""
        try:
            print("🔄 Migrating from classification to verification...")
            
            # Process dataset to get encodings and labels
            encodings, labels = dataset_processor.process_dataset()
            
            if len(encodings) == 0:
                print("❌ No encodings found to migrate")
                return False
            
            # Group encodings by student
            student_groups = {}
            for encoding, label in zip(encodings, labels):
                if label not in student_groups:
                    student_groups[label] = []
                student_groups[label].append(encoding)
            
            print(f"📊 Found {len(student_groups)} students to migrate:")
            for student_name, student_encodings in student_groups.items():
                print(f"   - {student_name}: {len(student_encodings)} encodings")
            
            # Add each student to verification database
            migration_count = 0
            for student_name, student_encodings in student_groups.items():
                student_id = f"student_{student_name.lower().replace(' ', '_')}"
                
                success = self.add_student(
                    student_id=student_id,
                    student_name=student_name,
                    nim=f"NIM_{student_name.upper()}",
                    kelas="Migrated_Class",
                    encodings=student_encodings
                )
                
                if success:
                    migration_count += 1
            
            print(f"✅ Migration completed: {migration_count}/{len(student_groups)} students")
            return migration_count > 0
            
        except Exception as e:
            print(f"❌ Migration error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def debug_verification(self, input_encoding, student_name=None):
        """Debug verification process step by step"""
        print(f"\n🔍 DEBUGGING VERIFICATION PROCESS")
        print("=" * 50)
        
        if student_name:
            print(f"Expected student: {student_name}")
        
        print(f"Input encoding shape: {input_encoding.shape}")
        print(f"Input encoding norm: {np.linalg.norm(input_encoding):.6f}")
        print(f"Database students: {len(self.student_encodings)}")
        print(f"Verification threshold: {self.verification_threshold}")
        
        # Run verification
        student_id, predicted_name, similarity, top_matches = self.verify_face(input_encoding, top_k=3)
        
        print(f"\n📊 VERIFICATION RESULTS:")
        print(f"   Predicted: {predicted_name}")
        print(f"   Similarity: {similarity:.6f}")
        print(f"   Verified: {similarity >= self.verification_threshold if similarity else False}")
        
        if student_name:
            is_correct = predicted_name == student_name
            print(f"   Correct: {'✅' if is_correct else '❌'}")
        
        print(f"\n🏆 TOP MATCHES:")
        for i, match in enumerate(top_matches[:3]):
            print(f"   {i+1}. {match['student_name']}: {match['max_similarity']:.6f}")
        
        return student_id, predicted_name, similarity, top_matches