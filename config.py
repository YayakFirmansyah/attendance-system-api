# config.py - SIMPLE VERSION
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Face detection settings
    MIN_FACE_SIZE = int(os.getenv('MIN_FACE_SIZE', 40))
    FACE_CONFIDENCE_THRESHOLD = float(os.getenv('FACE_CONFIDENCE_THRESHOLD', 0.9))
    
    # Recognition settings
    RECOGNITION_THRESHOLD = float(os.getenv('RECOGNITION_THRESHOLD', 0.7))
    
    # Paths
    MODEL_DIR = os.getenv('MODEL_DIR', 'models')
    
    # FaceNet settings
    FACENET_INPUT_SIZE = (160, 160)  # Standard FaceNet input
    
    # Ensure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)