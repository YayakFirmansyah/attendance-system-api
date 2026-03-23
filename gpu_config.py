# gpu_config.py - Konfigurasi TensorFlow untuk GPU NVIDIA
import tensorflow as tf
import os

def configure_gpu():
    """Konfigurasi TensorFlow untuk menggunakan GPU NVIDIA"""
    
    print("\n" + "="*60)
    print("🎮 KONFIGURASI GPU NVIDIA")
    print("="*60)
    
    # Cek GPU yang tersedia
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Set memory growth untuk mencegah TensorFlow menggunakan semua VRAM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Gunakan hanya GPU pertama (bisa diubah sesuai kebutuhan)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            print(f"✅ GPU Terdeteksi: {len(gpus)} GPU")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            
            print(f"✅ Menggunakan: {gpus[0].name}")
            print(f"✅ Memory Growth: Enabled (dinamis)")
            
            # Verifikasi TensorFlow build dengan CUDA
            print(f"\n📊 TensorFlow Version: {tf.__version__}")
            print(f"🔧 Built with CUDA: {tf.test.is_built_with_cuda()}")
            print(f"🎯 GPU Available: {tf.test.is_gpu_available(cuda_only=True)}")
            
            # Test sederhana
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"\n✅ Test GPU berhasil!")
                print(f"   Device: {c.device}")
            
            print("="*60 + "\n")
            return True
            
        except RuntimeError as e:
            print(f"❌ Error konfigurasi GPU: {e}")
            print("="*60 + "\n")
            return False
    else:
        print("❌ Tidak ada GPU yang terdeteksi!")
        print("⚠️  Pastikan:")
        print("   1. Driver NVIDIA sudah terinstall")
        print("   2. CUDA Toolkit terinstall (untuk TF 2.10: CUDA 11.2)")
        print("   3. cuDNN terinstall (untuk TF 2.10: cuDNN 8.1)")
        print("="*60 + "\n")
        return False

def get_gpu_info():
    """Tampilkan informasi GPU"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print("\n💻 INFORMASI GPU:")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"  Details: {gpu_details}")
            except:
                pass
    
    # Tampilkan TensorFlow info
    print(f"\nTensorFlow Version: {tf.__version__}")
    print(f"CUDA Available: {tf.test.is_built_with_cuda()}")

if __name__ == "__main__":
    # Test konfigurasi GPU
    configure_gpu()
    get_gpu_info()
