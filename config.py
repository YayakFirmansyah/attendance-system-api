import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'attendance_system')
    
    # Face Recognition
    FACE_CONFIDENCE_THRESHOLD = 0.7
    RECOGNITION_THRESHOLD = 0.6
    IMAGE_SIZE = (160, 160)
    
    # Paths
    DATASET_PATH = 'dataset'
    UPLOADS_PATH = 'uploads'
    MODELS_PATH = 'models'
    
    # Flask
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000