# README_GPU_SETUP.md

# 🎮 Setup Face Attendance System dengan GPU NVIDIA

## Persyaratan GPU

Untuk menggunakan TensorFlow dengan GPU NVIDIA, Anda memerlukan:

### 1. **NVIDIA Driver**

- Driver NVIDIA terbaru untuk GPU Anda
- Download: https://www.nvidia.com/download/index.aspx

### 2. **CUDA Toolkit 11.2**

- TensorFlow 2.10 membutuhkan CUDA 11.2
- Download: https://developer.nvidia.com/cuda-11.2.0-download-archive
- **PENTING**: Install di lokasi default yang disarankan

### 3. **cuDNN 8.1**

- Download dari: https://developer.nvidia.com/cudnn
- Extract dan copy file ke folder CUDA:
  - Copy `bin` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
  - Copy `include` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include`
  - Copy `lib` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib`

## Langkah-langkah Setup

### Langkah 1: Buat Virtual Environment

```bash
# Sudah dibuat di F:\face-attendace-project\attendance-system-api\venv
# Tidak ada file yang akan ditambahkan ke drive C
```

### Langkah 2: Install Dependencies dengan GPU Support

```bash
# Double-click file ini atau jalankan di terminal:
install_dependencies.bat
```

Script ini akan:

- Mengaktifkan virtual environment
- Upgrade pip
- Install semua package dari requirements.txt (termasuk tensorflow-gpu)
- Verifikasi instalasi GPU

### Langkah 3: Jalankan Flask Server

```bash
# Double-click file ini atau jalankan di terminal:
run_flask.bat
```

## Perintah Manual (Alternatif)

Jika ingin menjalankan secara manual:

```bash
# 1. Aktifkan virtual environment
venv\Scripts\activate

# 2. Install dependencies (jika belum)
pip install -r requirements.txt

# 3. Test GPU
python gpu_config.py

# 4. Jalankan Flask
python app.py
```

## Verifikasi GPU

Untuk memastikan GPU terdeteksi:

```bash
# Aktifkan venv terlebih dahulu
venv\Scripts\activate

# Jalankan test GPU
python gpu_config.py
```

Output yang diharapkan:

```
🎮 KONFIGURASI GPU NVIDIA
✅ GPU Terdeteksi: 1 GPU
   GPU 0: /physical_device:GPU:0
✅ Menggunakan: /physical_device:GPU:0
✅ Memory Growth: Enabled (dinamis)
```

## Troubleshooting

### ❌ GPU Tidak Terdeteksi

1. **Cek NVIDIA Driver:**

   ```bash
   nvidia-smi
   ```

   Jika error, install/update driver NVIDIA

2. **Cek CUDA:**

   ```bash
   nvcc --version
   ```

   Harus menunjukkan CUDA 11.2

3. **Cek Environment Variables:**
   Pastikan ada di System PATH:
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp`

### ⚠️ TensorFlow Masih Menggunakan CPU

Jika TensorFlow masih menggunakan CPU:

1. Uninstall TensorFlow biasa:

   ```bash
   pip uninstall tensorflow
   ```

2. Install TensorFlow GPU:

   ```bash
   pip install tensorflow-gpu==2.10.0
   ```

3. Restart terminal dan jalankan lagi

## Struktur Project

```
f:\face-attendace-project\attendance-system-api\
├── venv/                    # Virtual environment (di drive F)
├── app.py                   # Flask app (dengan GPU support)
├── gpu_config.py            # Konfigurasi GPU
├── requirements.txt         # Dependencies (tensorflow-gpu)
├── run_flask.bat           # Script jalankan Flask
├── install_dependencies.bat # Script install dependencies
├── models/
├── utils/
└── temp/
```

## Keuntungan Menggunakan GPU

- ⚡ **Inference lebih cepat**: 5-10x lebih cepat daripada CPU
- 🎯 **Real-time detection**: Cocok untuk video streaming
- 💪 **Handling multiple requests**: Bisa handle lebih banyak request bersamaan

## Port Default

Flask berjalan di:

- http://localhost:5000
- http://127.0.0.1:5000

## Menghentikan Server

Tekan `Ctrl + C` di terminal untuk menghentikan Flask server.
