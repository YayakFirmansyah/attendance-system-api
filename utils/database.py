# attendance-api-new/utils/database.py
import mysql.connector
from config import Config
from datetime import datetime

class Database:
    def __init__(self):
        self.connection = None
        
    def connect(self):
        """Connect to database"""
        try:
            self.connection = mysql.connector.connect(
                host=Config.DB_HOST,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                database=Config.DB_NAME,
                autocommit=True
            )
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False
    
    def get_students(self):
        """Get all active students"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM students WHERE status = 'active'")
            return cursor.fetchall()
        except Exception as e:
            print(f"Get students error: {e}")
            return []
    
    def save_attendance(self, student_id, class_id, confidence):
        """Save attendance record"""
        try:
            cursor = self.connection.cursor()
            
            # Check existing attendance
            check_query = """
            SELECT id FROM attendances 
            WHERE student_id = %s AND class_id = %s AND date = %s
            """
            cursor.execute(check_query, (student_id, class_id, datetime.now().date()))
            
            if cursor.fetchone():
                print("Attendance already recorded today")
                return False
            
            # Insert new attendance
            insert_query = """
            INSERT INTO attendances (student_id, class_id, date, check_in, status, created_at, updated_at)
            VALUES (%s, %s, %s, %s, 'present', %s, %s)
            """
            now = datetime.now()
            cursor.execute(insert_query, (
                student_id, class_id, now.date(), now.time(), now, now
            ))
            
            # Log attendance
            log_query = """
            INSERT INTO attendance_logs (student_id, class_id, timestamp, confidence_score, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(log_query, (
                student_id, class_id, now, confidence, now, now
            ))
            
            print(f"✓ Attendance saved for student {student_id}")
            return True
            
        except Exception as e:
            print(f"Save attendance error: {e}")
            return False

# Global instance
db = Database()