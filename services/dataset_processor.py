import os
import cv2
import numpy as np
from utils.face_detector import FaceDetector
from utils.face_encoder import FaceNetEncoder
from config import Config
# import random
# import matplotlib.pyplot as plt
# from collections import defaultdict

class EnhancedDatasetProcessor:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_encoder = FaceNetEncoder()
        self.stats = {
            'total_images': 0,
            'faces_detected': 0,
            'faces_encoded': 0,
            'students_processed': 0,
            'failed_images': []
        }
        print("✓ Enhanced dataset processor initialized")

    def process_dataset(self, dataset_path=None):
        """Process dataset with detailed statistics"""
        if dataset_path is None:
            dataset_path = Config.DATASET_PATH

        print(f"Processing dataset from: {dataset_path}")
        print("=" * 50)

        encodings = []
        labels = []
        self.stats = {
            'total_images': 0,
            'faces_detected': 0,
            'faces_encoded': 0,
            'students_processed': 0,
            'failed_images': []
        }

        student_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        student_folders.sort()
        print(f"Found {len(student_folders)} student folders")

        for student_name in student_folders:
            student_path = os.path.join(dataset_path, student_name)
            print(f"\nProcessing: {student_name}")
            print("-" * 30)

            image_files = [f for f in os.listdir(student_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            if len(image_files) == 0:
                print(f"  ⚠️  No images found for {student_name}")
                continue

            student_encodings = []

            for image_file in image_files:
                image_path = os.path.join(student_path, image_file)
                self.stats['total_images'] += 1
                print(f"  📷 Processing: {image_file}")

                encoding = self.process_image(image_path, student_name)

                if encoding is not None:
                    student_encodings.append(encoding)
                    encodings.append(encoding)
                    labels.append(student_name)
                    self.stats['faces_encoded'] += 1
                    print(f"     ✅ Encoded successfully")
                else:
                    self.stats['failed_images'].append(image_path)
                    print(f"     ❌ Failed to process")

            if len(student_encodings) > 0:
                self.stats['students_processed'] += 1
                print(f"  📊 Generated {len(student_encodings)} encodings for {student_name}")
            else:
                print(f"  ⚠️  No valid encodings for {student_name}")

        self.print_statistics()
        return encodings, labels

    def process_image(self, image_path, student_name):
        """Process single image with detailed logging"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"     ❌ Failed to load image")
                return None

            print(f"     📐 Image size: {image.shape}")

            faces = self.face_detector.detect_faces(image)
            self.stats['faces_detected'] += len(faces)

            if not faces:
                print(f"     👤 No faces detected")
                return None

            if len(faces) > 1:
                print(f"     👥 Multiple faces detected ({len(faces)}), using best one")

            best_face = max(faces, key=lambda x: x['confidence'])
            print(f"     🎯 Face confidence: {best_face['confidence']:.3f}")

            face = self.face_detector.extract_face(image, best_face)
            if face is None:
                print(f"     ❌ Failed to extract face")
                return None

            print(f"     ✂️  Extracted face size: {face.shape}")

            encoding = self.face_encoder.encode_face(face)
            if encoding is not None:
                print(f"     🧠 Generated encoding: {encoding.shape}")
                return encoding
            else:
                print(f"     ❌ Failed to generate encoding")
                return None

        except Exception as e:
            print(f"     ❌ Error processing image: {e}")
            return None

    def print_statistics(self):
        """Print processing statistics"""
        print("\n" + "=" * 50)
        print("DATASET PROCESSING STATISTICS")
        print("=" * 50)
        print(f"📁 Total images processed: {self.stats['total_images']}")
        print(f"👤 Faces detected: {self.stats['faces_detected']}")
        print(f"🧠 Faces encoded: {self.stats['faces_encoded']}")
        print(f"👥 Students processed: {self.stats['students_processed']}")

        success_rate = (self.stats['faces_encoded'] / self.stats['total_images']) * 100 if self.stats['total_images'] > 0 else 0
        print(f"✅ Success rate: {success_rate:.1f}%")

        if self.stats['failed_images']:
            print(f"\n❌ Failed images ({len(self.stats['failed_images'])}):")
            for failed_img in self.stats['failed_images'][:5]:
                print(f"   - {failed_img}")
            if len(self.stats['failed_images']) > 5:
                print(f"   ... and {len(self.stats['failed_images']) - 5} more")
        print("=" * 50)

    def validate_dataset(self, dataset_path=None):
        """Validate dataset structure before processing"""
        if dataset_path is None:
            dataset_path = Config.DATASET_PATH

        print(f"Validating dataset structure: {dataset_path}")

        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path does not exist: {dataset_path}")
            return False

        student_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

        if len(student_folders) < 2:
            print(f"❌ Need at least 2 student folders, found {len(student_folders)}")
            return False

        valid_students = 0
        total_images = 0

        for student_name in student_folders:
            student_path = os.path.join(dataset_path, student_name)
            image_files = [f for f in os.listdir(student_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            if len(image_files) >= 2:
                valid_students += 1
                total_images += len(image_files)
                print(f"✅ {student_name}: {len(image_files)} images")
            else:
                print(f"⚠️  {student_name}: {len(image_files)} images (need at least 2)")

        print(f"\n📊 Summary:")
        print(f"   Valid students: {valid_students}/{len(student_folders)}")
        print(f"   Total images: {total_images}")

        if valid_students < 2:
            print(f"❌ Need at least 2 students with 2+ images each")
            return False

        print(f"✅ Dataset validation passed")
        return True

    def analyze_dataset_quality(self, dataset_path=None):
        """Analyze dataset quality in detail"""
        if dataset_path is None:
            dataset_path = Config.DATASET_PATH

        print(f"\n🔍 ANALYZING DATASET QUALITY: {dataset_path}")
        print("=" * 60)

        quality_report = {
            'students': {},
            'summary': {
                'total_students': 0,
                'total_images': 0,
                'avg_quality_score': 0
            }
        }

        student_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

        for student_name in student_folders:
            print(f"\n📊 Analyzing: {student_name}")
            print("-" * 30)

            student_path = os.path.join(dataset_path, student_name)
            image_files = [f for f in os.listdir(student_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            student_metrics = {
                'total_images': len(image_files),
                'valid_faces': 0,
                'face_confidences': [],
                'face_sizes': [],
                'brightness_levels': [],
                'blur_scores': [],
                'problems': []
            }

            for img_file in image_files:
                img_path = os.path.join(student_path, img_file)
                metrics = self.analyze_single_image(img_path, img_file)

                if metrics:
                    if metrics['face_detected']:
                        student_metrics['valid_faces'] += 1
                        student_metrics['face_confidences'].append(metrics['face_confidence'])
                        student_metrics['face_sizes'].append(metrics['face_size'])
                        student_metrics['brightness_levels'].append(metrics['brightness'])
                        student_metrics['blur_scores'].append(metrics['blur_score'])
                    else:
                        student_metrics['problems'].append(f"{img_file}: No face detected")

                    if metrics['too_dark']:
                        student_metrics['problems'].append(f"{img_file}: Too dark")
                    if metrics.get('too_bright', False):
                        student_metrics['problems'].append(f"{img_file}: Too bright")
                    if metrics['too_blurry']:
                        student_metrics['problems'].append(f"{img_file}: Too blurry")
                    if metrics['face_too_small']:
                        student_metrics['problems'].append(f"{img_file}: Face too small")

            quality_score = self.calculate_quality_score(student_metrics)
            student_metrics['quality_score'] = quality_score

            self.print_student_analysis(student_name, student_metrics)

            quality_report['students'][student_name] = student_metrics
            quality_report['summary']['total_students'] += 1
            quality_report['summary']['total_images'] += student_metrics['total_images']

        if quality_report['summary']['total_students'] > 0:
            avg_quality = sum([metrics['quality_score'] for metrics in quality_report['students'].values()]) / quality_report['summary']['total_students']
            quality_report['summary']['avg_quality_score'] = avg_quality

        self.print_quality_summary(quality_report)
        return quality_report

    def analyze_single_image(self, image_path, filename):
        """Analyze a single image in detail"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            metrics = {
                'filename': filename,
                'face_detected': False,
                'face_confidence': 0.0,
                'face_size': 0.0,
                'brightness': 0.0,
                'blur_score': 0.0,
                'too_dark': False,
                'too_bright': False,
                'too_blurry': False,
                'face_too_small': False
            }

            faces = self.face_detector.detect_faces(image)
            if faces:
                best_face = max(faces, key=lambda x: x['confidence'])
                metrics['face_detected'] = True
                metrics['face_confidence'] = best_face['confidence']
                box = best_face['box']
                face_area = box[2] * box[3]
                img_area = image.shape[0] * image.shape[1]
                metrics['face_size'] = face_area / img_area
                if metrics['face_size'] < 0.05:
                    metrics['face_too_small'] = True

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            metrics['brightness'] = brightness
            if brightness < 50:
                metrics['too_dark'] = True
            elif brightness > 200:
                metrics['too_bright'] = True

            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['blur_score'] = blur_score
            if blur_score < 100:
                metrics['too_blurry'] = True

            return metrics

        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            return None

    def calculate_quality_score(self, student_metrics):
        """Calculate quality score (0-100)"""
        score = 0
        if student_metrics['total_images'] == 0:
            return 0

        valid_ratio = student_metrics['valid_faces'] / student_metrics['total_images']
        score += valid_ratio * 25

        if student_metrics['valid_faces'] == 0:
            return score

        avg_confidence = np.mean(student_metrics['face_confidences'])
        if avg_confidence > 0.9:
            score += 25
        elif avg_confidence > 0.8:
            score += 20
        elif avg_confidence > 0.7:
            score += 15
        elif avg_confidence > 0.6:
            score += 10

        avg_face_size = np.mean(student_metrics['face_sizes'])
        if avg_face_size > 0.15:
            score += 20
        elif avg_face_size > 0.1:
            score += 15
        elif avg_face_size > 0.05:
            score += 10

        brightness_std = np.std(student_metrics['brightness_levels'])
        if brightness_std < 20:
            score += 15
        elif brightness_std < 40:
            score += 10
        elif brightness_std < 60:
            score += 5

        avg_blur = np.mean(student_metrics['blur_scores'])
        if avg_blur > 500:
            score += 15
        elif avg_blur > 200:
            score += 12
        elif avg_blur > 100:
            score += 8
        elif avg_blur > 50:
            score += 5

        return min(100, score)

    def print_student_analysis(self, student_name, metrics):
        """Display analysis for a single student"""
        print(f"👤 {student_name.upper()}")
        print(f"   📁 Total images: {metrics['total_images']}")
        print(f"   ✅ Faces detected: {metrics['valid_faces']}")
        if metrics['valid_faces'] > 0:
            print(f"   🎯 Avg confidence: {np.mean(metrics['face_confidences']):.3f}")
            print(f"   📏 Avg face size: {np.mean(metrics['face_sizes']):.3f}")
            print(f"   💡 Avg brightness: {np.mean(metrics['brightness_levels']):.1f}")
            print(f"   🔍 Avg sharpness: {np.mean(metrics['blur_scores']):.1f}")
        print(f"   📊 Quality Score: {metrics['quality_score']:.1f}/100")
        if metrics['problems']:
            print(f"   ⚠️  Issues found:")
            for problem in metrics['problems'][:3]:
                print(f"      - {problem}")
            if len(metrics['problems']) > 3:
                print(f"      ... and {len(metrics['problems']) - 3} more issues")

    def print_quality_summary(self, quality_report):
        """Display dataset quality summary and recommendations"""
        print(f"\n" + "=" * 60)
        print("📋 DATASET QUALITY SUMMARY")
        print("=" * 60)
        summary = quality_report['summary']
        print(f"👥 Total students: {summary['total_students']}")
        print(f"📸 Total images: {summary['total_images']}")
        print(f"📊 Avg quality score: {summary['avg_quality_score']:.1f}/100")

        print(f"\n📈 QUALITY COMPARISON:")
        for student_name, metrics in quality_report['students'].items():
            status = "🟢 GOOD" if metrics['quality_score'] >= 70 else "🟡 FAIR" if metrics['quality_score'] >= 50 else "🔴 POOR"
            print(f"   {student_name}: {metrics['quality_score']:.1f}/100 {status}")

        print(f"\n💡 RECOMMENDATIONS:")
        avg_quality = summary['avg_quality_score']
        if avg_quality < 50:
            print("🚨 POOR DATASET QUALITY - Major improvements needed:")
            print("   1. Use good and consistent lighting")
            print("   2. Ensure faces fill at least 10% of the image area")
            print("   3. Use high-resolution cameras")
            print("   4. Avoid blurry or dark images")
            print("   5. Take photos from various angles (front, slight left/right)")
        elif avg_quality < 70:
            print("⚠️  FAIR DATASET QUALITY - Some improvements recommended:")
            print("   1. Improve low-quality images")
            print("   2. Add pose and lighting variety")
            print("   3. Ensure all images are sharp and bright")
        else:
            print("✅ GOOD DATASET QUALITY - Ready for training!")

        for student_name, metrics in quality_report['students'].items():
            if metrics['quality_score'] < 60:
                print(f"\n🔧 Recommendations for {student_name}:")
                if metrics['valid_faces'] / metrics['total_images'] < 0.8:
                    print("   - Take clearer face photos")
                if metrics['face_confidences'] and np.mean(metrics['face_confidences']) < 0.8:
                    print("   - Improve face angle and lighting")
                if metrics['face_sizes'] and np.mean(metrics['face_sizes']) < 0.05:
                    print("   - Take closer (face close-up) photos")
                if metrics['blur_scores'] and np.mean(metrics['blur_scores']) < 100:
                    print("   - Use sharper images")

    def validate_and_analyze_dataset(self, dataset_path=None):
        """Combined validation and quality analysis"""
        print("🚀 STARTING FULL DATASET ANALYSIS...")
        is_valid = self.validate_dataset(dataset_path)
        if not is_valid:
            return False, None
        quality_report = self.analyze_dataset_quality(dataset_path)
        avg_quality = quality_report['summary']['avg_quality_score']
        print(f"\n🎯 ANALYSIS CONCLUSION:")
        if avg_quality >= 70:
            print("✅ Dataset ready for training with good performance!")
            return True, quality_report
        elif avg_quality >= 50:
            print("⚠️  Dataset usable, but improvements recommended")
            return True, quality_report
        else:
            print("❌ Dataset needs improvement before training!")
            return False, quality_report
