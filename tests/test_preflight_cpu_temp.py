#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_preflight_cpu_temp.py
#
# The Pi goes into a sealed box in a filter house. Thermal throttling
# looks exactly like "inference got slower for no reason", so preflight
# must name it. This pins the verdict: temperature bands and the
# vcgencmd get_throttled bits (current vs since-boot).
#
# Run with pytest, or directly:
#     python tests/test_preflight_cpu_temp.py
# ============================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preflight import cpu_temp_verdict  # noqa: E402


def test_cool_and_never_throttled_passes():
    assert cpu_temp_verdict(52.0, 0x0) == ('pass', [], [])
    assert cpu_temp_verdict(52.0, None)[0] == 'pass'


def test_warm_box_warns():
    assert cpu_temp_verdict(72.5, 0x0)[0] == 'warn'


def test_throttling_temperature_fails():
    assert cpu_temp_verdict(81.0, 0x0)[0] == 'fail'


def test_throttled_now_fails_even_when_cool():
    level, now, _ = cpu_temp_verdict(60.0, 0x4)
    assert level == 'fail' and now == ['throttled']


def test_under_voltage_now_fails():
    level, now, _ = cpu_temp_verdict(45.0, 0x1)
    assert level == 'fail' and 'under-voltage' in now


def test_throttled_since_boot_warns():
    """0x50000: under-voltage and throttling occurred earlier, not now — the
    exact signature of a hot afternoon that has since cooled."""
    level, now, past = cpu_temp_verdict(55.0, 0x50000)
    assert level == 'warn' and now == []
    assert set(past) == {'under-voltage', 'throttled'}


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
