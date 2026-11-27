#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_bulk_label.py
#
# Bulk "no heron" is the main way the negative training data this project
# lacks gets made — so it gets the same guards as a single label, not a
# shortcut round them:
#
#   * each clip's own recorded frame type picks its domain; a day clip and an
#     IR clip in one batch land in different trees
#   * a bulk action never picks a domain: clips caught mid-switch, with no
#     capture-time record, or whose filed frame is a transition frame are
#     left in triage and reported
#   * 'heron' is never a bulk label (it needs a hand-drawn box)
#   * exactly one frame per clip is filed: frame 0, the thumbnail the
#     operator looked at
#   * undo deletes exactly what the batch wrote, and spares a clip that was
#     relabelled by hand since
#
# Run with pytest, or directly:
#     python tests/test_bulk_label.py
# ============================================================

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.media_store import MediaStore  # noqa: E402


def _clip(path: Path, n=5, w=640, h=480):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))
    for i in range(n):
        writer.write(np.full((h, w, 3), 40 * i, np.uint8))   # frame 0 is black
    writer.release()
    return str(path)


def _harness(tmp: Path):
    """Flask client over a fake system: six clips covering every routing case."""
    clips = tmp / 'clips'
    clips.mkdir()

    def row(i, frame_type, types, labelled=None):
        meta = json.dumps({'clip_frame_types': types}) if types else '{}'
        return {'id': i, 'video_path': _clip(clips / f'c{i}.mp4'), 'frame_type': frame_type,
                'metadata': meta, 'manually_labeled': 1 if labelled else 0,
                'user_label': labelled}

    rows = {
        1: row(1, 'day', ['day'] * 5),                        # -> labeled/day
        2: row(2, 'ir', ['ir'] * 5),                          # -> labeled/ir
        3: row(3, 'establishing', ['day'] * 5),               # unsettled: needs a human
        4: row(4, 'day', None),                               # no record: needs a human
        5: row(5, 'day', ['ir', 'day', 'day', 'day', 'day']),  # frame 0 is a transition frame
        6: row(6, 'day', ['day'] * 5, labelled='cat'),        # already labelled
    }

    class _DB:
        db_path = tmp / 'data' / 'heron.db'

        def get_detection(self, i):
            return dict(rows[i]) if i in rows else None

        def update_manual_label(self, i, label):
            rows[i]['manually_labeled'], rows[i]['user_label'] = 1, label

        def clear_manual_label(self, i, expected):
            if rows[i]['manually_labeled'] and rows[i]['user_label'] == expected:
                rows[i]['manually_labeled'], rows[i]['user_label'] = 0, None
                return True
            return False

    _DB.db_path.parent.mkdir()

    class _System:
        database = _DB()
        media_store = MediaStore(str(tmp / 'media'))

    from ui import app as ui_app
    ui_app.init_app(_System())
    return ui_app.app.test_client(), rows, tmp / 'media' / 'labeled'


def _files(root: Path):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()) \
        if root.exists() else []


def test_each_clip_lands_in_its_own_domain_and_the_rest_stay_in_triage():
    with tempfile.TemporaryDirectory() as t:
        client, rows, labeled = _harness(Path(t))
        r = client.post('/api/label/bulk', json={'detection_ids': [1, 2, 3, 4, 5, 6],
                                                 'label': 'background'})
        body = r.get_json()
        assert r.status_code == 200 and body['success'], body
        assert body['filed'] == {'day': 1, 'ir': 1}, body['filed']
        assert {x['id'] for x in body['left']} == {3, 4, 5, 6}
        reasons = {x['id']: x['reason'] for x in body['left']}
        assert 'already labelled' in reasons[6]
        assert 'transition' in reasons[5]
        # day and IR in separate trees, one image + one EMPTY label each
        files = _files(labeled)
        assert sum(f.startswith('day/images/') for f in files) == 1
        assert sum(f.startswith('ir/images/') for f in files) == 1
        assert not any(f.startswith('dark/') for f in files)
        for txt in labeled.rglob('*.txt'):
            assert txt.read_text() == '', f"{txt} is not a background example"
        # rows: 1 and 2 labelled background; 3-5 untouched, still in triage
        assert rows[1]['user_label'] == rows[2]['user_label'] == 'background'
        assert not any(rows[i]['manually_labeled'] for i in (3, 4, 5))
        assert rows[6]['user_label'] == 'cat'


def test_a_clip_whose_files_cannot_be_flushed_stays_in_triage():
    """storage/durable.py: label files must reach the card before the row
    says 'labelled'. If one clip's cannot, that clip stays in triage and the
    rest of the batch is filed and undoable — the batch is not aborted."""
    import unittest.mock as mock
    from storage import media_store
    with tempfile.TemporaryDirectory() as t:
        client, rows, labeled = _harness(Path(t))
        real = media_store.make_durable

        def fails_for_ir(*paths):
            if any('/ir/' in str(p) for p in paths):
                raise OSError('EIO')
            return real(*paths)
        with mock.patch.object(media_store, 'make_durable', side_effect=fails_for_ir):
            r = client.post('/api/label/bulk', json={'detection_ids': [1, 2],
                                                     'label': 'background'})
        body = r.get_json()
        assert r.status_code == 200 and body['filed'] == {'day': 1}, body
        assert [x['id'] for x in body['left']] == [2]
        assert 'could not save' in body['left'][0]['reason']
        assert rows[1]['user_label'] == 'background' and not rows[2]['manually_labeled']
        assert body['batch_id']


def test_the_filed_frame_is_frame_zero():
    """One frame per clip, and it is the thumbnail frame — the only one the
    operator looked at. Frame 0 of the test clip is black; frame 2 is 80."""
    with tempfile.TemporaryDirectory() as t:
        client, _, labeled = _harness(Path(t))
        client.post('/api/label/bulk', json={'detection_ids': [1], 'label': 'background'})
        images = list(labeled.rglob('*.jpg'))
        assert len(images) == 1
        assert cv2.imread(str(images[0])).mean() < 10


def test_heron_is_never_a_bulk_label():
    with tempfile.TemporaryDirectory() as t:
        client, rows, labeled = _harness(Path(t))
        for label in ('heron', 'unknown', 'cat', None):
            r = client.post('/api/label/bulk', json={'detection_ids': [1], 'label': label})
            assert r.status_code == 400, (label, r.get_json())
        assert _files(labeled) == [] and not rows[1]['manually_labeled']


def test_bad_id_lists_are_refused():
    with tempfile.TemporaryDirectory() as t:
        client, _, _ = _harness(Path(t))
        for ids in ([], None, 'abc', list(range(501))):
            r = client.post('/api/label/bulk', json={'detection_ids': ids, 'label': 'background'})
            assert r.status_code == 400, ids


def test_preview_reports_the_split_and_writes_nothing():
    with tempfile.TemporaryDirectory() as t:
        client, rows, labeled = _harness(Path(t))
        r = client.post('/api/label/bulk', json={'detection_ids': [1, 2, 3, 4, 5],
                                                 'label': 'background', 'preview': True})
        body = r.get_json()
        assert body['preview'] and body['filed'] == {'day': 1, 'ir': 1}
        assert {x['id'] for x in body['left']} == {3, 4, 5}
        assert body['batch_id'] is None
        assert _files(labeled) == []
        assert not any(rows[i]['manually_labeled'] for i in (1, 2))


def test_duplicate_ids_are_filed_once():
    with tempfile.TemporaryDirectory() as t:
        client, _, labeled = _harness(Path(t))
        r = client.post('/api/label/bulk', json={'detection_ids': [1, 1, 1],
                                                 'label': 'background'})
        assert r.get_json()['filed'] == {'day': 1}
        assert len(list(labeled.rglob('*.jpg'))) == 1


def test_undo_removes_exactly_the_batch_and_restores_triage():
    with tempfile.TemporaryDirectory() as t:
        client, rows, labeled = _harness(Path(t))
        # a hand label from before the batch must survive the undo
        client.post('/api/label', json={'detection_id': 4, 'label': 'background',
                                        'frame_index': 0, 'domain': 'day'})
        before = _files(labeled)
        assert len(before) == 2
        batch = client.post('/api/label/bulk', json={'detection_ids': [1, 2],
                                                     'label': 'background'}).get_json()
        assert len(_files(labeled)) == 6
        assert client.post('/api/label/bulk/undo',
                           json={'batch_id': 'wrong'}).status_code == 409
        r = client.post('/api/label/bulk/undo', json={'batch_id': batch['batch_id']})
        body = r.get_json()
        assert r.status_code == 200 and body['restored'] == 2 and body['files_removed'] == 4
        assert _files(labeled) == before
        assert not rows[1]['manually_labeled'] and not rows[2]['manually_labeled']
        assert rows[4]['user_label'] == 'background'
        assert client.post('/api/label/bulk/undo',
                           json={'batch_id': batch['batch_id']}).status_code == 404


def test_undo_spares_a_clip_relabelled_since():
    with tempfile.TemporaryDirectory() as t:
        client, rows, labeled = _harness(Path(t))
        batch = client.post('/api/label/bulk', json={'detection_ids': [1, 2],
                                                     'label': 'background'}).get_json()
        rows[2]['user_label'] = 'bird'            # relabelled by hand afterwards
        body = client.post('/api/label/bulk/undo',
                           json={'batch_id': batch['batch_id']}).get_json()
        assert body['restored'] == 1 and body['kept'] == [2]
        assert rows[2]['user_label'] == 'bird'
        assert len(list((labeled / 'ir').rglob('*.jpg'))) == 1   # clip 2's file kept


def test_single_background_label_takes_the_same_path():
    """The modal's 'no heron' goes through the same core: same domain rule,
    same empty annotation."""
    with tempfile.TemporaryDirectory() as t:
        client, rows, labeled = _harness(Path(t))
        r = client.post('/api/label', json={'detection_id': 2, 'label': 'background',
                                            'frame_index': 0})
        assert r.get_json() == {'success': True, 'domain': 'ir'}
        assert [p.read_text() for p in labeled.rglob('*.txt')] == ['']
        r = client.post('/api/label', json={'detection_id': 3, 'label': 'background',
                                            'frame_index': 0})
        assert r.status_code == 409 and not rows[3]['manually_labeled']


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
