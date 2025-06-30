# attendance-system-api/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Face Recognition Thresholds
    RECOGNITION_THRESHOLD = float(os.getenv('RECOGNITION_THRESHOLD', 0.75))
    FACE_CONFIDENCE_THRESHOLD = float(os.getenv('FACE_CONFIDENCE_THRESHOLD', 0.9))
    
    # Image Processing
    IMAGE_SIZE = (160, 160)  # Standard FaceNet input size
    
    # Paths
    DATASET_PATH = os.getenv('DATASET_PATH', 'dataset')
    MODEL_PATH = os.getenv('MODEL_PATH', 'models')
    
    # Database
    VERIFICATION_DB_PATH = os.path.join(MODEL_PATH, 'verification_database.pkl')
    
    # Ensure directories exist
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)