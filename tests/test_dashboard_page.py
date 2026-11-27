#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_dashboard_page.py
#
# Static checks on the dashboard page itself. The Pi's dashboard is used
# on a LAN that may have no internet, and the brief for it was "no external
# dependencies. None." — no CDN, no web fonts, no build step. So:
#   * nothing the page loads may come from another host;
#   * every icon the page or its script uses must exist in the inline
#     sprite, or it renders as an empty box with no error anywhere;
#   * no emoji: they render differently (or not at all) per platform,
#     which is why the icons are drawn as SVG.
#
# Run with pytest, or directly:
#     python tests/test_dashboard_page.py
# ============================================================

import re
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / 'ui' / 'templates' / 'index.html'


def _page():
    return PAGE.read_text(encoding='utf-8')


def test_nothing_is_loaded_from_another_host():
    src = _page()
    # The SVG namespace is an identifier, not a request.
    src = src.replace('xmlns="http://www.w3.org/2000/svg"', '')
    offenders = re.findall(r'''(?:src|href|action)\s*=\s*["'](?:https?:)?//[^"']+''', src)
    offenders += re.findall(r'''url\(\s*["']?(?:https?:)?//[^)]+\)''', src)
    offenders += re.findall(r'@import', src)
    offenders += re.findall(r'''fetch\(\s*["'`](?:https?:)?//''', src)
    assert not offenders, offenders


def test_every_icon_used_is_in_the_sprite():
    src = _page()
    defined = set(re.findall(r'<symbol id="i-([a-z-]+)"', src))
    used = set(re.findall(r'href="#i-([a-z-]+)"', src))
    used |= set(re.findall(r"icon\('([a-z-]+)'", src))
    # toast() maps its type to a name held in a table
    toast_table = re.search(r"const icons = \{([^}]*)\}", src).group(1)
    used |= set(re.findall(r"'([a-z-]+)'", toast_table))
    alert_table = re.search(r"const glyph = icon\(([^;]*)\);", src).group(1)
    used |= set(re.findall(r"'([a-z-]+)'", alert_table)) - {'critical', 'warning', 'motion'}
    missing = used - defined
    assert not missing, f"icons used but not in the sprite: {sorted(missing)}"
    assert len(defined) >= 10


def test_no_emoji():
    emoji = re.compile('[\U0001F300-\U0001FAFF☀-➿️]')
    hits = [(i, line.strip()[:80]) for i, line in enumerate(_page().split('\n'), 1)
            if emoji.search(line)]
    assert not hits, hits


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
