# test_threshold.py - Test dengan threshold rendah untuk SVM model
import requests
import base64
import json
import os

def test_yayak_detection():
    """Test khusus untuk deteksi Yayak dengan threshold rendah"""
    
    print("🎯 TESTING YAYAK DETECTION dengan THRESHOLD RENDAH")
    print("="*60)
    
    # Ambil foto test (ganti dengan path foto Yayak)
    image_path = input("Masukkan path foto Yayak untuk test: ").strip()
    
    if not image_path:
        print("❌ No image path provided")
        return
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        print("💡 Make sure the file path is correct")
        return
        
    # Check file size
    file_size = os.path.getsize(image_path)
    print(f"📁 File: {image_path}")
    print(f"📊 Size: {file_size:,} bytes")
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        print("⚠️ File too large (>10MB), might cause timeout")
    
    try:
        print("📷 Reading image...")
        # Convert to base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            base64_image = f"data:image/jpeg;base64,{image_data}"
        
        print(f"✅ Image converted to base64 ({len(base64_image):,} characters)")
        
        # Test API health first
        print("\n🔍 Checking API health...")
        try:
            health_response = requests.get("http://localhost:5000/api/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                print("✅ API is healthy")
                if health_data.get('model_info'):
                    print(f"   Model loaded: {health_data['model_info'].get('model_loaded')}")
                    print(f"   Classes: {health_data['model_info'].get('classes', [])}")
            else:
                print(f"❌ API health check failed: {health_response.status_code}")
                return
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            print("💡 Make sure Flask app is running: python app.py")
            return
        
        # Test face verification
        print("\n🚀 Sending image for face verification...")
        print("⏳ This may take 10-30 seconds...")
        
        response = requests.post(
            "http://localhost:5000/api/verify-face",
            json={"image": base64_image},
            headers={"Content-Type": "application/json"},
            timeout=60  # Increase timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n📊 HASIL DETEKSI:")
            print(f"Success: {result.get('success')}")
            print(f"Message: {result.get('message')}")
            print(f"Total faces: {result.get('total_faces')}")
            print(f"Verified faces: {result.get('verified_faces')}")
            
            if result.get('results'):
                for i, face in enumerate(result.get('results', [])):
                    name = face.get('predicted_name')
                    confidence = face.get('recognition_confidence')
                    verified = face.get('verified')
                    
                    print(f"\n👤 FACE {i+1} RESULT:")
                    print(f"   Name: {name}")
                    print(f"   Confidence: {confidence:.3f}")
                    print(f"   Verified: {'✅ YES' if verified else '❌ NO'}")
                    
                    if verified:
                        print(f"   🎉 SUCCESS! {name} detected with confidence {confidence:.3f}")
                    else:
                        print(f"   ⚠️ Not verified (confidence too low or unknown)")
                    
                    print(f"\n📈 ALL PREDICTIONS:")
                    all_preds = face.get('all_predictions', {})
                    for pred_name, pred_conf in sorted(all_preds.items(), key=lambda x: x[1], reverse=True):
                        marker = "👑" if pred_name == name else "  "
                        print(f"   {marker} {pred_name}: {pred_conf:.3f}")
                    
                    # Analysis
                    sorted_preds = sorted(all_preds.items(), key=lambda x: x[1], reverse=True)
                    if len(sorted_preds) >= 2:
                        gap = sorted_preds[0][1] - sorted_preds[1][1]
                        print(f"\n🔍 CONFIDENCE ANALYSIS:")
                        print(f"   Best: {sorted_preds[0][0]} ({sorted_preds[0][1]:.3f})")
                        print(f"   Second: {sorted_preds[1][0]} ({sorted_preds[1][1]:.3f})")
                        print(f"   Gap: {gap:.3f}")
                        print(f"   Gap >= 0.03: {'✅' if gap >= 0.03 else '❌'}")
                        print(f"   Confidence >= 0.08: {'✅' if confidence >= 0.08 else '❌'}")
                        
                        if gap >= 0.03 and confidence >= 0.08:
                            print("   🎯 SHOULD BE VERIFIED!")
                        else:
                            print("   ⚠️ Threshold not met")
            else:
                print("\n❌ No faces detected in image")
                print("💡 Make sure:")
                print("   - Face is clearly visible")
                print("   - Good lighting")
                print("   - Face not too small/far")
                
        else:
            print(f"\n❌ API Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error message: {error_data.get('message')}")
            except:
                print(f"Response: {response.text[:500]}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout! API is too slow.")
        print("💡 Try:")
        print("   - Smaller image file")
        print("   - Restart Flask app")
        print("   - Check GPU/CPU performance")
    except Exception as e:
        print(f"❌ Test error: {e}")

def quick_test_with_current_dir():
    """Quick test dengan foto di current directory"""
    print("\n🔍 Looking for images in current directory...")
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = []
    
    for file in os.listdir('.'):
        if file.lower().endswith(image_extensions):
            images.append(file)
    
    if images:
        print(f"Found {len(images)} image(s):")
        for i, img in enumerate(images):
            print(f"   {i+1}. {img}")
        
        choice = input("\nEnter number to test (or press Enter to skip): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(images):
            selected_image = images[int(choice)-1]
            print(f"\n🎯 Testing with: {selected_image}")
            
            # Test the selected image
            import sys
            sys.argv = ['test_threshold.py']  # Reset argv
            global image_path
            image_path = selected_image
            
            # Run test directly
            test_image_direct(selected_image)
    else:
        print("No images found in current directory")

def test_image_direct(image_path):
    """Test image langsung tanpa input"""
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
        
    print(f"📷 Testing: {image_path}")
    
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            base64_image = f"data:image/jpeg;base64,{image_data}"
        
        response = requests.post(
            "http://localhost:5000/api/verify-face",
            json={"image": base64_image},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Result: {result.get('message')}")
            for face in result.get('results', []):
                name = face.get('predicted_name')
                confidence = face.get('recognition_confidence')
                verified = face.get('verified')
                print(f"   👤 {name} ({confidence:.3f}) {'✅' if verified else '❌'}")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Check for command line argument
    import sys
    if len(sys.argv) > 1:
        test_image_direct(sys.argv[1])
    else:
        # Interactive mode
        quick_test_with_current_dir()
        test_yayak_detection()