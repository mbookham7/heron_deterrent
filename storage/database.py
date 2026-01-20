# ============================================================
# FILE: storage/database.py
# ============================================================

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

class Database:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # Detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                label TEXT NOT NULL,
                confidence REAL,
                video_path TEXT,
                metadata TEXT,
                manually_labeled BOOLEAN DEFAULT 0,
                user_label TEXT
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                type TEXT NOT NULL,
                detection_id INTEGER,
                message TEXT,
                acknowledged BOOLEAN DEFAULT 0,
                FOREIGN KEY (detection_id) REFERENCES detections(id)
            )
        ''')
        
        # SMS log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sms_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                alert_id INTEGER,
                success BOOLEAN,
                FOREIGN KEY (alert_id) REFERENCES alerts(id)
            )
        ''')
        
        self.conn.commit()
    
    def save_detection(self, label: str, confidence: float, 
                      video_path: str, metadata: Dict[str, Any]) -> int:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO detections (label, confidence, video_path, metadata)
            VALUES (?, ?, ?, ?)
        ''', (label, confidence, video_path, json.dumps(metadata)))
        self.conn.commit()
        return cursor.lastrowid
    
    def save_alert(self, alert_type: str, detection_id: Optional[int], 
                   message: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (type, detection_id, message)
            VALUES (?, ?, ?)
        ''', (alert_type, detection_id, message))
        self.conn.commit()
        return cursor.lastrowid
    
    def log_sms(self, alert_id: int, success: bool):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO sms_log (alert_id, success)
            VALUES (?, ?)
        ''', (alert_id, success))
        self.conn.commit()
    
    def get_unclassified_detections(self) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM detections 
            WHERE label = 'unknown' AND manually_labeled = 0
            ORDER BY timestamp DESC
        ''')
        return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM alerts 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_detections(self, label: Optional[str] = None, 
                      limit: int = 100) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        if label:
            cursor.execute('''
                SELECT * FROM detections 
                WHERE label = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (label, limit))
        else:
            cursor.execute('''
                SELECT * FROM detections 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def update_manual_label(self, detection_id: int, user_label: str):
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE detections 
            SET manually_labeled = 1, user_label = ?
            WHERE id = ?
        ''', (user_label, detection_id))
        self.conn.commit()
    
    def cleanup_old_records(self, days: int):
        cursor = self.conn.cursor()
        cutoff_date = datetime.now() - timedelta(days=days)
        cursor.execute('''
            DELETE FROM detections 
            WHERE timestamp < ?
        ''', (cutoff_date,))
        cursor.execute('''
            DELETE FROM alerts 
            WHERE timestamp < ?
        ''', (cutoff_date,))
        self.conn.commit()
    
    def get_last_sms_time(self) -> Optional[datetime]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MAX(timestamp) as last_time FROM sms_log
            WHERE success = 1
        ''')
        row = cursor.fetchone()
        if row and row['last_time']:
            return datetime.fromisoformat(row['last_time'])
        return None
    
    def close(self):
        if self.conn:
            self.conn.close()
