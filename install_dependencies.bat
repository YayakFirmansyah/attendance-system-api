@echo off
REM install_dependencies.bat - Install semua dependencies dengan GPU support

echo ========================================
echo   INSTALASI DEPENDENCIES + GPU SUPPORT
echo ========================================
echo.

REM Aktifkan virtual environment
echo [1/5] Mengaktifkan Virtual Environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo.
    echo [ERROR] Virtual environment belum dibuat!
    echo Jalankan perintah berikut terlebih dahulu:
    echo   python -m venv venv
    pause
    exit /b 1
)

echo [OK] Virtual Environment aktif
echo.

REM Upgrade pip
echo [2/5] Upgrade pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel

echo.
echo [3/5] Install dependencies dari requirements.txt...
pip install -r requirements.txt

echo.
echo [4/5] Verifikasi instalasi TensorFlow GPU...
python -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__); print('GPU Available:', tf.config.list_physical_devices('GPU'))"

echo.
echo [5/5] Test GPU Configuration...
python gpu_config.py

echo.
echo ========================================
echo   INSTALASI SELESAI!
echo ========================================
echo.
echo CATATAN PENTING:
echo - Pastikan NVIDIA Driver sudah terinstall
echo - Untuk TensorFlow 2.10, butuh:
echo   * CUDA 11.2
echo   * cuDNN 8.1
echo.
echo Download dari:
echo - CUDA: https://developer.nvidia.com/cuda-11.2.0-download-archive
echo - cuDNN: https://developer.nvidia.com/cudnn
echo.
pause
