# Cross-repo file sharing between `heron_deterrent` and `heron-detector`

**tl;dr — vendored copies, with parity tests. Both repos ship byte-identical
copies of two files; parity tests catch drift. Do NOT invent a fourth mechanism.**

## What is shared

Two files must produce the SAME pixels in the deterrent app's inference path
and in the detector's evaluation harness, or measured recall does not match
what the deployed device actually sees:

| canonical (this repo)         | vendored copy (heron-detector) |
|-------------------------------|--------------------------------|
| `tools/camera_sim.py`         | `src/camera_sim.py`            |
| `ai/letterbox.py`             | `src/letterbox.py`             |

`camera_sim` lives under `tools/` here, not under `ai/`, because the deployed
device has a camera — it never simulates one. Nothing in `tools/` ships to
the Coral or the Pi.

`letterbox` lives under `ai/` because it IS part of the runtime path — the
detector on-device calls it on every frame.

## Why not import from the sibling directly

`heron-detector` runs on EC2 for headless evaluation. `heron_deterrent` is
NOT checked out there. Any of:

- `sys.path.insert('../heron_deterrent/...')` — silently fails on EC2
- installing `heron_deterrent` as a package — makes the app carry the eval
  harness's tail on the Pi, wrong direction
- a symlink from `heron-detector/src/camera_sim.py` to the sibling — becomes
  a dangling link on EC2

...breaks the EC2 path, or the on-device path, or both.

## Why not just duplicate quietly

The three prior sessions produced three different answers to this problem:

- `letterbox` — duplicate with a parity test (kept)
- `camera_sim` (first attempt) — sibling-path import via `sys.path.insert`
- `camera_sim` (second attempt) — duplicate verbatim, parity test promised
  but never written, and a header comment saying it would be "copied
  verbatim to heron-detector when the second run happens" that would have
  overwritten the richer copy with the thinner one

Silent drift between the two copies is exactly the class of bug that
produced the 19/44-vs-18/44 recall discrepancy on `image94.jpg` — one
copy used `INTER_AREA` on the upsample path, the other `INTER_LINEAR`,
and no test caught it.

## The rule

1. Canonical copies live HERE (`heron_deterrent`).
2. `heron-detector` vendors byte-identical copies at `src/camera_sim.py`
   and `src/letterbox.py`, both committed to that repo.
3. Parity tests in `heron-detector` (`tests/test_camera_sim_parity.py`,
   `tests/test_letterbox_parity.py`) fail if any copy drifts. The tests
   skip gracefully with an explicit message when the sibling isn't
   checked out — they do NOT silently pass.
4. To edit either file: edit HERE, then in `heron-detector` run
   `scripts/sync_from_heron_deterrent.sh` and re-run BOTH repos' tests.
5. If the parity test fails, fix the divergence. Do not silence the test.

## What to do when this is not enough

If a future change needs a third shared file, extend the same
mechanism — add the file to the sync script and add a parity test.
Do not introduce a fourth strategy.

If the shared surface grows to more than a few files, or the app repo
starts wanting to import from `heron-detector`, then the vendored-copy
approach is the wrong shape and this doc's assumption needs revisiting.
Until then, live with the ugliness — it is explicit, testable, and
survives EC2.
