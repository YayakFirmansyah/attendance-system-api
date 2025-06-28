# services/migration_service.py
from utils.scalable_face_recognizer import ScalableFaceRecognizer
from services.dataset_processor import EnhancedDatasetProcessor

class MigrationService:
    def __init__(self):
        self.scalable_recognizer = ScalableFaceRecognizer()
        self.dataset_processor = EnhancedDatasetProcessor()
    
    def migrate_to_verification(self):
        """Migrate from classification to verification approach"""
        try:
            print("🔄 STARTING MIGRATION TO FACE VERIFICATION")
            print("=" * 60)
            
            # Load existing database if any
            self.scalable_recognizer.load_database()
            
            # Process current dataset
            print("📂 Processing dataset...")
            encodings, labels = self.dataset_processor.process_dataset()
            
            if len(encodings) == 0:
                return {
                    'success': False,
                    'message': 'No encodings found in dataset'
                }
            
            # Group encodings by student
            student_groups = {}
            for encoding, label in zip(encodings, labels):
                if label not in student_groups:
                    student_groups[label] = []
                student_groups[label].append(encoding)
            
            print(f"👥 Found {len(student_groups)} students:")
            for student_name, encodings_list in student_groups.items():
                print(f"   - {student_name}: {len(encodings_list)} encodings")
            
            # Add each student to verification database
            migration_results = {}
            
            for student_name, student_encodings in student_groups.items():
                student_id = f"student_{student_name.lower().replace(' ', '_')}"
                
                success = self.scalable_recognizer.add_student(
                    student_id=student_id,
                    student_name=student_name,
                    nim=f"NIM_{student_name.upper()}",
                    kelas="Class_2024",
                    encodings=student_encodings
                )
                
                migration_results[student_name] = {
                    'success': success,
                    'student_id': student_id,
                    'encoding_count': len(student_encodings)
                }
            
            # Get final statistics
            stats = self.scalable_recognizer.get_statistics()
            
            print("\n✅ MIGRATION COMPLETED")
            print(f"📊 Total students: {stats['total_students']}")
            print(f"🧠 Total encodings: {stats['total_encodings']}")
            print(f"📈 Avg encodings per student: {stats['avg_encodings_per_student']:.1f}")
            
            return {
                'success': True,
                'message': f'Migration completed: {stats["total_students"]} students',
                'statistics': stats,
                'migration_details': migration_results
            }
            
        except Exception as e:
            print(f"❌ Migration error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Migration error: {str(e)}'
            }
    
    def test_verification_system(self):
        """Test the verification system with current dataset"""
        try:
            print("🧪 TESTING VERIFICATION SYSTEM")
            print("=" * 50)
            
            # Process dataset for testing
            encodings, labels = self.dataset_processor.process_dataset()
            
            if len(encodings) == 0:
                return {
                    'success': False,
                    'message': 'No test data available'
                }
            
            # Run benchmark
            benchmark_results = self.scalable_recognizer.benchmark_verification(
                encodings, labels
            )
            
            print(f"📊 VERIFICATION RESULTS:")
            print(f"   Accuracy: {benchmark_results['accuracy']:.1%}")
            print(f"   Precision: {benchmark_results['precision']:.1%}")
            print(f"   Recall: {benchmark_results['recall']:.1%}")
            print(f"   F1-Score: {benchmark_results['f1_score']:.1%}")
            
            return {
                'success': True,
                'message': f'Verification test completed - {benchmark_results["accuracy"]:.1%} accuracy',
                'benchmark_results': benchmark_results
            }
            
        except Exception as e:
            print(f"❌ Test error: {e}")
            return {
                'success': False,
                'message': f'Test error: {str(e)}'
            }