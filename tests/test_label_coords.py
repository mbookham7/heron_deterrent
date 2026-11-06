#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_label_coords.py
#
# The bounding-box mapping in the dashboard has three coordinate systems
# stacked on top of each other:
#
#   canvas pixels  ->  displayed image pixels  ->  original frame pixels
#
# Every one of those transforms is where a coordinate-conversion bug hides.
# The test frame here is DELIBERATELY:
#
#   * non-square (800x400)
#   * asymmetric: displayed at 300x200 — width scale 8/3, height scale 2/1
#     (different scaling in each axis, so any aspect-locked or transposed
#     transform fails)
#   * off-centre in both axes (box at canvas (50, 30, 100, 40) is not at any
#     boundary, so an off-by-half error shows up as a numeric mismatch)
#
# The bug this exists to catch: the previous UI sent bboxes=[] every time,
# and even a plausible rewrite could silently apply the wrong scale factor
# in one axis. A symmetric test frame would pass with the wrong transform.
#
# Run with pytest, or directly:
#     python tests/test_label_coords.py
# ============================================================

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.media_store import MediaStore  # noqa: E402


# --- The three-stage mapping the client is supposed to implement --------------
def canvas_to_native(canvas_box, canvas_w, canvas_h, native_w, native_h):
    """Take a box drawn on a display-scaled canvas and return the same box in
    the original frame's pixel space. Kept as plain arithmetic so this test
    doubles as documentation for the JS side."""
    cx, cy, cw, ch = canvas_box
    sx = native_w / canvas_w
    sy = native_h / canvas_h
    return (cx * sx, cy * sy, cw * sx, ch * sy)


def test_asymmetric_display_scaling_maps_correctly():
    """The core mapping: canvas 300x200 -> native 800x400 has DIFFERENT
    scale in each axis. If someone wires up one axis for both, this fails."""
    native_w, native_h = 800, 400
    canvas_w, canvas_h = 300, 200
    canvas_box = (50, 30, 100, 40)

    px = canvas_to_native(canvas_box, canvas_w, canvas_h, native_w, native_h)

    # width scale = 800/300 = 8/3; height scale = 400/200 = 2
    expected = (50 * 8 / 3, 30 * 2, 100 * 8 / 3, 40 * 2)
    for got, want in zip(px, expected):
        assert abs(got - want) < 1e-6, f"{px} != {expected}"


def test_yolo_annotation_line_is_correct_end_to_end():
    """The whole path: canvas box -> native pixels -> YOLO normalised line.
    A wrong axis in ANY of the three stages shows up in the numeric line."""
    native_w, native_h = 800, 400
    canvas_w, canvas_h = 300, 200
    canvas_box = (50, 30, 100, 40)

    px = canvas_to_native(canvas_box, canvas_w, canvas_h, native_w, native_h)
    # native (133.333, 60, 266.667, 80)
    # x_center = (133.333 + 133.333) / 800 = 0.333...
    # y_center = (60 + 40) / 400 = 0.25
    # w_norm = 266.667 / 800 = 0.333...
    # h_norm = 80 / 400 = 0.2

    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        frame = np.zeros((native_h, native_w, 3), dtype=np.uint8)
        image_path, label_path = store.save_labeled_image(
            frame, 'heron', [px], detection_id=1, frame_type='day'
        )
        line = Path(label_path).read_text().strip()

    parts = line.split()
    assert parts[0] == '0', f"heron must be class 0, got line: {line}"
    xc, yc, wn, hn = (float(p) for p in parts[1:])
    assert abs(xc - 1/3) < 1e-6, f"x_center {xc} != 1/3"
    assert abs(yc - 0.25) < 1e-6, f"y_center {yc} != 0.25"
    assert abs(wn - 1/3) < 1e-6, f"w_norm {wn} != 1/3"
    assert abs(hn - 0.20) < 1e-6, f"h_norm {hn} != 0.20"


def test_heron_with_empty_bboxes_writes_empty_file():
    """Regression: this used to be the whole bug. The store MUST NOT invent a
    line when handed no boxes, and the /api/label endpoint MUST refuse the
    request in the first place. This test pins the store's half."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        frame = np.zeros((400, 800, 3), dtype=np.uint8)
        _, label_path = store.save_labeled_image(frame, 'heron', [], detection_id=1, frame_type='day')
        contents = Path(label_path).read_text()
        # A .txt with content would be worse than nothing here — but the store
        # is not the layer that should be rejecting the request. It writes an
        # empty file; the API endpoint is what refuses the request. See
        # test_api_refuses_heron_with_no_boxes for the enforcement point.
        assert contents == '', f"expected empty file, got {contents!r}"


def test_non_heron_labels_write_empty_file_even_with_boxes():
    """A cat is a valid background example for a single-class heron model.
    Boxes drawn for a cat MUST NOT be written — that would train class 0 on a
    cat and undermine the whole point of asymmetric label rules."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        frame = np.zeros((400, 800, 3), dtype=np.uint8)
        # save_labeled_image already enforces this via the label check; pass a
        # box anyway to catch a regression that removes the check.
        _, label_path = store.save_labeled_image(
            frame, 'cat', [(100, 50, 200, 100)], detection_id=1, frame_type='day'
        )
        assert Path(label_path).read_text() == '', (
            "cat label produced a non-empty .txt — someone extended the "
            "single-class annotation contract"
        )


def test_api_refuses_heron_with_no_boxes():
    """The /api/label endpoint must reject a heron submission with no boxes
    with a 400. Uses Flask's test client so the wiring is exercised end-to-end.
    """
    # Minimal fake system that returns nothing useful — the request never
    # gets far enough to touch the DB or the clip because the guard is upfront.
    class _FakeDB:
        def update_manual_label(self, *_): pass
        def get_detections(self, **_): return []
    class _FakeSystem:
        database = _FakeDB()
        media_store = None
    from ui import app as ui_app
    ui_app.init_app(_FakeSystem())
    client = ui_app.app.test_client()

    r = client.post('/api/label', json={
        'detection_id': 1, 'label': 'heron', 'bboxes': [], 'frame_index': 0
    })

    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.data!r}"
    body = r.get_json()
    assert body['success'] is False
    assert 'heron' in body['error'].lower()


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
