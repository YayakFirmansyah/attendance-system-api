@echo off
REM run_flask.bat - Script untuk menjalankan Flask dengan Virtual Environment

echo ========================================
echo   FACE ATTENDANCE SYSTEM - FLASK API
echo ========================================
echo.

REM Aktifkan virtual environment
echo [1/3] Mengaktifkan Virtual Environment...
call venv\Scripts\activate.bat

REM Cek apakah aktivasi berhasil
if errorlevel 1 (
    echo.
    echo [ERROR] Gagal mengaktifkan virtual environment!
    echo Pastikan virtual environment sudah dibuat dengan: python -m venv venv
    pause
    exit /b 1
)

echo [OK] Virtual Environment aktif
echo.

REM Set environment variables untuk Flask
echo [2/3] Mengatur Flask Environment...
set FLASK_APP=app.py
set FLASK_ENV=development
set FLASK_DEBUG=1

echo [OK] Flask Environment siap
echo.

REM Jalankan Flask
echo [3/3] Menjalankan Flask Server...
echo.
echo ========================================
echo   Server akan berjalan di:
echo   http://127.0.0.1:5000
echo   http://localhost:5000
echo ========================================
echo.
echo Tekan Ctrl+C untuk menghentikan server
echo.

python app.py

REM Jika Flask berhenti
echo.
echo Server dihentikan.
pause
