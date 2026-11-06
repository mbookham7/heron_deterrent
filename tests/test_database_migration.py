#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_database_migration.py
#
# Guards the schema migration for the `bbox` column. The column exists so we
# stop throwing the model's box away — model-assisted labelling later needs it.
#
# Two cases matter:
#   1. Fresh DB — CREATE TABLE includes bbox from the start.
#   2. Pre-existing DB (the deployed device) — ALTER TABLE ADD COLUMN runs and
#      leaves existing rows with bbox = NULL. Adding a column with a default
#      other than NULL on a big SQLite table is O(n); NULL is O(1). So this test
#      also documents the choice: existing rows keep NULL forever, they are not
#      backfilled.
#
# Run with pytest, or directly:
#     python tests/test_database_migration.py
# ============================================================

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.database import Database  # noqa: E402


# Schema before the bbox column was added.
LEGACY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    label TEXT NOT NULL,
    confidence REAL,
    video_path TEXT,
    metadata TEXT,
    manually_labeled BOOLEAN DEFAULT 0,
    user_label TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    type TEXT NOT NULL,
    detection_id INTEGER,
    message TEXT,
    acknowledged BOOLEAN DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sms_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    alert_id INTEGER,
    success BOOLEAN
);
"""


def _cols(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_fresh_db_has_bbox_column():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / 'heron.db'
        db = Database(str(db_path))
        cols = _cols(db.conn, 'detections')
        assert 'bbox' in cols, f"CREATE TABLE missed bbox: {cols}"
        db.close()


def test_migration_adds_bbox_to_pre_existing_db():
    """Simulate the deployed device: DB was created by the old code, then the
    process starts on the new code. Migration must add the column and existing
    rows must keep their data (with bbox = NULL)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / 'heron.db'

        # Bootstrap a legacy DB WITHOUT the bbox column.
        conn = sqlite3.connect(db_path)
        conn.executescript(LEGACY_SCHEMA_SQL)
        conn.execute(
            "INSERT INTO detections (label, confidence, video_path, metadata) "
            "VALUES ('heron', 0.87, '/tmp/x.mp4', '{}')"
        )
        conn.commit()
        conn.close()

        # Open with the new Database — migration should run.
        db = Database(str(db_path))
        cols = _cols(db.conn, 'detections')
        assert 'bbox' in cols, "migration failed to add bbox column"

        # Existing rows preserved, bbox is NULL.
        row = db.conn.execute(
            "SELECT label, confidence, bbox FROM detections WHERE label = 'heron'"
        ).fetchone()
        assert row is not None, "existing row disappeared during migration"
        assert row[0] == 'heron' and abs(row[1] - 0.87) < 1e-6
        assert row[2] is None, f"existing rows should be NULL, got {row[2]!r}"
        db.close()


def test_migration_is_idempotent():
    """Opening the DB twice must not fail. Guards against a naive
    ALTER TABLE ADD COLUMN with no prior check."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / 'heron.db'
        Database(str(db_path)).close()
        # Second open runs migration again; if this raises, migration is
        # not idempotent.
        Database(str(db_path)).close()


def test_save_detection_persists_bbox():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(str(Path(tmp) / 'heron.db'))
        bbox = (0.1, 0.2, 0.3, 0.4)
        det_id = db.save_detection(
            label='heron', confidence=0.91, video_path='/tmp/x.mp4',
            metadata={'type': 'CRITICAL'}, bbox=bbox, frame_type='day',
        )
        row = db.conn.execute(
            "SELECT bbox FROM detections WHERE id = ?", (det_id,)
        ).fetchone()
        stored = json.loads(row[0])
        for got, want in zip(stored, bbox):
            assert abs(got - want) < 1e-9, f"{stored} != {bbox}"
        db.close()


def test_save_detection_without_bbox_stores_null():
    """Backwards-compat: save_detection was called without bbox originally
    and the arg is optional. Should end up as SQL NULL, not the string 'None'
    or 'null'."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(str(Path(tmp) / 'heron.db'))
        det_id = db.save_detection(
            label='unknown', confidence=0.5, video_path='',
            metadata={'type': 'WARNING'}, frame_type='day',
        )
        row = db.conn.execute(
            "SELECT bbox FROM detections WHERE id = ?", (det_id,)
        ).fetchone()
        assert row[0] is None, f"expected NULL, got {row[0]!r}"
        db.close()


# --------------------------------------------------------------------------
# frame_type column. Same migration pattern as bbox; existing rows stay NULL
# rather than being backfilled as 'day' — the old 06:00 start was dark half
# the year, so 'day' would be a guess.
# --------------------------------------------------------------------------

def test_migration_adds_frame_type_to_bbox_era_db():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / 'heron.db'
        conn = sqlite3.connect(db_path)
        conn.executescript(LEGACY_SCHEMA_SQL)
        conn.execute('ALTER TABLE detections ADD COLUMN bbox TEXT')
        conn.execute("INSERT INTO detections (label, confidence) VALUES ('heron', 0.9)")
        conn.commit()
        conn.close()

        db = Database(str(db_path))
        assert 'frame_type' in _cols(db.conn, 'detections')
        row = db.conn.execute("SELECT label, frame_type FROM detections").fetchone()
        assert row[0] == 'heron' and row[1] is None, f"legacy row changed: {tuple(row)}"
        db.close()
        Database(str(db_path)).close()  # idempotent


def test_save_detection_requires_frame_type():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(str(Path(tmp) / 'heron.db'))
        try:
            db.save_detection(label='heron', confidence=0.9, video_path='', metadata={})
        except TypeError:
            pass
        else:
            raise AssertionError("save_detection accepted a row with no frame_type")
        db.close()


def test_frame_type_filter_and_motion_stats():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(str(Path(tmp) / 'heron.db'))
        db.save_detection('heron', 0.97, '', {}, frame_type='day')
        db.save_detection('motion', None, '', {}, frame_type='ir')
        db.save_detection('motion', None, '', {}, frame_type='switching')
        db.conn.execute("INSERT INTO detections (label, confidence) VALUES ('unknown', 0.1)")
        db.conn.commit()

        assert [d['label'] for d in db.get_detections(frame_type='ir')] == ['motion']
        assert {d['frame_type'] for d in db.get_detections(frame_type='unknown')} == {None, 'switching'}

        stats = db.get_stats()
        assert stats['detections_total'] == 2, "motion rows must not count as detections"
        assert stats['motion_total'] == 2
        assert stats['herons_today'] == 1

        queue = {d['label'] for d in db.get_unclassified_detections()}
        assert 'motion' in queue, "night motion clips must reach the triage queue"
        db.close()


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
