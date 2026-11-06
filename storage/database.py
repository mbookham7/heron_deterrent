# ============================================================
# FILE: storage/database.py
# ============================================================

import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

class Database:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = None
        self._initialize_db()

    def _initialize_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._migrate()

    def _create_tables(self):
        with self._lock:
            cursor = self.conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    label TEXT NOT NULL,
                    confidence REAL,
                    video_path TEXT,
                    metadata TEXT,
                    manually_labeled BOOLEAN DEFAULT 0,
                    user_label TEXT,
                    bbox TEXT,
                    frame_type TEXT
                )
            ''')

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

    def _migrate(self):
        """Apply any schema changes needed for a database created by an older
        version. Idempotent — a fresh DB has the columns via CREATE TABLE above.

        `bbox` keeps the model's box once we have it, instead of recomputing
        every time (or, worse, throwing it away and needing manual annotation
        later).

        `frame_type` records which image domain the clip came from: day, ir or
        dark when the mode was settled, establishing or switching when it was
        not. Rows written before this column existed stay NULL — they are NOT
        backfilled as 'day', because the old 06:00 start was already in the
        dark for half the year. NULL means "unknown", and the labelling path
        classifies the actual frame rather than trusting a guess."""
        with self._lock:
            cursor = self.conn.cursor()
            cols = {row['name'] for row in cursor.execute('PRAGMA table_info(detections)')}
            # SQLite's ALTER TABLE is limited but ADD COLUMN is safe and
            # constant-time (metadata-only) since NULL is the default.
            for name in ('bbox', 'frame_type'):
                if name not in cols:
                    cursor.execute(f'ALTER TABLE detections ADD COLUMN {name} TEXT')
            self.conn.commit()

    def save_detection(self, label: str, confidence: Optional[float],
                      video_path: str, metadata: Dict[str, Any],
                      bbox: Optional[tuple] = None, *, frame_type: str) -> int:
        """Persist a detection. `bbox` is (x, y, w, h) normalised 0-1 against
        the ORIGINAL frame — same units as Detection.bbox — or None. Stored as
        JSON so downstream queries can pull it out with json_extract().

        `frame_type` is keyword-only and required: a row with no image domain
        is how day and night data end up mixed. `confidence` is None for a
        motion-only row — no model ran, so there is no score to store.
        """
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO detections (label, confidence, video_path, metadata, bbox, frame_type)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (label, confidence, video_path, json.dumps(metadata),
                  json.dumps(list(bbox)) if bbox is not None else None,
                  frame_type))
            self.conn.commit()
            return cursor.lastrowid

    def save_alert(self, alert_type: str, detection_id: Optional[int],
                   message: str) -> int:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (type, detection_id, message)
                VALUES (?, ?, ?)
            ''', (alert_type, detection_id, message))
            self.conn.commit()
            return cursor.lastrowid

    def log_sms(self, alert_id: int, success: bool):
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO sms_log (alert_id, success)
                VALUES (?, ?)
            ''', (alert_id, success))
            self.conn.commit()

    def get_unclassified_detections(self) -> List[Dict[str, Any]]:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM detections
                WHERE label IN ('unknown', 'motion') AND manually_labeled = 0
                ORDER BY timestamp DESC
                LIMIT 500
            ''')
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM alerts
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_detections(self, label: Optional[str] = None,
                      limit: int = 100,
                      frame_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Most recent detections, optionally filtered by label and frame type.

        frame_type='unknown' selects rows with no settled domain: legacy NULL
        rows and clips captured while establishing or switching.
        """
        where, args = [], []
        if label:
            where.append('label = ?')
            args.append(label)
        if frame_type == 'unknown':
            where.append("(frame_type IS NULL OR frame_type NOT IN ('day', 'ir', 'dark'))")
        elif frame_type:
            where.append('frame_type = ?')
            args.append(frame_type)
        sql = 'SELECT * FROM detections'
        if where:
            sql += ' WHERE ' + ' AND '.join(where)
        sql += ' ORDER BY timestamp DESC LIMIT ?'
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(sql, (*args, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_detection(self, detection_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute(
                'SELECT * FROM detections WHERE id = ?', (detection_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_stats(self) -> Dict[str, Any]:
        """Counts for the dashboard stats strip, computed here rather than by
        counting a capped list client-side.

        Motion-only rows are counted separately and never as detections: no
        model looked at them, so counting them alongside model output would
        imply something was identified. "Today" is the UTC date, matching how
        SQLite's CURRENT_TIMESTAMP stores the column.
        """
        with self._lock:
            row = self.conn.execute('''
                SELECT
                  COALESCE(SUM(label != 'motion'), 0)                               AS detections_total,
                  COALESCE(SUM(label = 'heron' AND date(timestamp) = date('now')), 0) AS herons_today,
                  COALESCE(SUM(label = 'motion'), 0)                                AS motion_total,
                  COALESCE(SUM(label = 'motion' AND date(timestamp) = date('now')), 0) AS motion_today,
                  MAX(CASE WHEN label = 'heron' THEN timestamp END)                 AS last_heron
                FROM detections
            ''').fetchone()
            alerts = self.conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE type != 'MOTION'"
            ).fetchone()[0]
            return {**dict(row), 'alerts_total': alerts}

    def update_manual_label(self, detection_id: int, user_label: str):
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE detections
                SET manually_labeled = 1, user_label = ?
                WHERE id = ?
            ''', (user_label, detection_id))
            self.conn.commit()

    def cleanup_old_records(self, days: int):
        with self._lock:
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
        with self._lock:
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
