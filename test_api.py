# test_api.py - Quick test untuk memastikan semua komponen berjalan
import sys
import os

def test_imports():
    """Test semua import yang dibutuhkan"""
    print("🧪 Testing imports...")
    
    try:
        from config import Config
        print("✅ Config imported")
        print(f"   IMAGE_SIZE: {Config.IMAGE_SIZE}")
        print(f"   RECOGNITION_THRESHOLD: {Config.RECOGNITION_THRESHOLD}")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from utils.face_detector import FaceDetector
        detector = FaceDetector()
        print("✅ FaceDetector imported and initialized")
    except Exception as e:
        print(f"❌ FaceDetector failed: {e}")
        return False
    
    try:
        from utils.face_encoder import FaceNetEncoder
        encoder = FaceNetEncoder()
        print("✅ FaceNetEncoder imported and initialized")
        print(f"   Model type: {encoder.model_type}")
    except Exception as e:
        print(f"❌ FaceNetEncoder failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🔬 Testing basic functionality...")
    
    try:
        import numpy as np
        from utils.face_encoder import FaceNetEncoder
        
        encoder = FaceNetEncoder()
        
        # Test dengan dummy image
        dummy_face = np.random.randint(0, 255, (*encoder.image_size, 3), dtype=np.uint8)
        encoding = encoder.encode_face(dummy_face)
        
        if encoding is not None:
            print(f"✅ Encoding test passed - dimension: {encoding.shape}")
            return True
        else:
            print("❌ Encoding test failed")
            return False
            
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TESTING FACE RECOGNITION COMPONENTS")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please fix the errors above.")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Functionality tests failed. Please check the implementation.")
        sys.exit(1)
    
    print("\n🎉 ALL TESTS PASSED!")
    print("You can now run: python app.py")