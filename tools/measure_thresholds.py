#!/usr/bin/env python3
# ============================================================
# FILE: tools/measure_thresholds.py
#
# Re-derives ai.deter_threshold and ai.alert_threshold from actual footage,
# so config.yaml stops carrying numbers no-one can reproduce. Also prints
# the exact YAML block to paste into config.
#
# ------------------------------------------------------------------
# GROUND-TRUTH PRINCIPLE — READ THIS BEFORE CHANGING DEFAULTS
# ------------------------------------------------------------------
#
# Directory names under media/ are the model's PREDICTIONS, never ground
# truth. media/heron/ contains clips the CURRENT MODEL called heron. It
# does not contain herons. On the deployed hardware today, every clip in
# media/ (both media/heron/ and media/unknown/) is bench footage of the
# same indoor scene — a person in a black polo shirt, no bird in frame at
# any point. That is why the ~0.9 scores on those clips are false
# positives worth measuring, not successes worth counting.
#
# Anyone reaching for media/heron/ as a positives set is reading the
# model's opinion of itself. Do not do this. Ground truth lives in
# directories a human curated:
#
#   * negatives — media/ (recursive, WITHOUT excluding media/heron/)
#     Every frame in there is a verified true negative today.
#   * positives — ../heron-data/heron_project/dataset/val/images/
#     44 real heron images. Same stock-photo domain as training, so they
#     flatter the model; they are provisional until field positives exist.
#
# WHY THIS EXISTS
#
# The old config.yaml claimed ai.confidence_threshold=0.75 was "clear of the
# ~0.70 spikes seen on non-heron frames". That number came from a hand
# measurement that no-one has been able to reproduce since. Repeat runs
# against real bench footage showed 18% of no-heron frames scoring above it,
# and the man-in-a-black-polo false positive at 0.886 is exactly what a
# single-threshold-at-0.75 policy allows through.
#
# This script is the reproducible replacement. If you disagree with the
# suggested thresholds, run it, argue about the numbers instead of the vibe.
#
# WHY THE OVERLAP GUARD EXISTS
#
# An earlier version of this script defaulted --positives to media/heron/
# and --negatives to media/, then silently subtracted the positives from
# the negatives to avoid double-counting. That silent subtraction hid two
# errors at once: 175 frames of the model's WORST false positives were
# being counted as positives, AND those same frames were being removed
# from the negatives pool. Both errors flattered the model. Now the two
# sets are required to be disjoint, and a non-empty intersection aborts
# the run rather than being silently reconciled.
#
# CAMERA-SIMULATION FIX
#
# An earlier version letterboxed 1536x2048 stock positives straight to the
# model's 640x640 input — the same size the letterbox produces from a 640x480
# webcam frame, but from ~10x more source pixels and without the 4:3 crop
# a real camera applies. Recall came out at 0.55 at deter=0.90; the honest
# camera-simulated number is 0.42. See ai/camera_sim.py for the full story.
# Positives are simulated by default; negatives are already camera-native
# and are NEVER simulated. --positives-source=camera skips simulation, for
# use with pond-labelled data as it arrives.
#
# USAGE
#
#   python tools/measure_thresholds.py
#   python tools/measure_thresholds.py --positives /path/to/heron_images \
#                                      --negatives /path/to/no_heron_clips \
#                                      --positives-source stock \
#                                      --sim-mode both \
#                                      --step 1 --k 3 --m 5
#
#   # Once you have field-labelled positives at native 640x480:
#   python tools/measure_thresholds.py --positives ./labelled_positives \
#                                      --positives-source camera
# ============================================================

import argparse
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.camera_sim import simulate_camera          # noqa: E402
from ai.detector import ObjectDetector             # noqa: E402
from ai.model_loader import ModelLoader            # noqa: E402
from utils.config_loader import Config             # noqa: E402

# Extract every Nth frame from each clip. Default 1 = every frame, because
# skipping frames systematically misses the tails of the score distribution,
# and the tail of the negatives distribution is exactly what governs how safe
# a threshold is. An earlier default of 5 gave the negatives max as 0.840 on
# footage whose true max is 0.894 — a 0.054 flattery gap that would tempt a
# reader into lowering the deter threshold. Pass --step >1 only if inference
# cost forces you to.
DEFAULT_STEP = 1

# Model-level threshold used only for finding the "best anchor per frame".
# We want the raw score distribution, not filtered detections.
SCORE_FLOOR = 0.05

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv'}


Frame = np.ndarray
Transform = Callable[[Frame], Frame]


def iter_frames(path: Path, step: int,
                transform: Optional[Transform] = None) -> Iterable[Frame]:
    """Yield frames from a file. Images: one frame. Videos: every `step`th.

    If `transform` is provided, it is applied to each frame BEFORE yielding.
    That is the point at which camera simulation runs for stock positives —
    at the frame boundary, so every downstream consumer (score_frame,
    frame_scores, per-clip aggregation) sees pixels that match what the
    device would have received. A raw-imread path here would silently
    inflate recall on high-resolution stock images.
    """
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        img = cv2.imread(str(path))
        if img is not None:
            yield transform(img) if transform is not None else img
        return

    if suffix not in VIDEO_SUFFIXES:
        return

    cap = cv2.VideoCapture(str(path))
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield transform(frame) if transform is not None else frame
            i += 1
    finally:
        cap.release()


def score_frame(detector: ObjectDetector, frame: np.ndarray) -> float:
    """Best anchor score for `frame`. Not filtered by the config threshold —
    this is a measurement, not a decision."""
    with detector._lock:
        input_data, _ = detector.preprocess_frame(frame)
        detector.interpreter.set_tensor(
            detector.input_details[0]['index'], input_data)
        detector.interpreter.invoke()
        out = detector.interpreter.get_tensor(detector.output_details[0]['index'])

    arr = detector.dequantize(out, detector.output_details[0])
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        return 0.0
    if arr.shape[0] == detector.num_features and arr.shape[1] != detector.num_features:
        arr = arr.T
    class_scores = arr[:, 4:]
    return float(class_scores.max())


def collect_scores(files: List[Path], detector: ObjectDetector,
                   step: int, label: str,
                   transform: Optional[Transform] = None) -> List[Tuple[Path, float]]:
    """Return (path, best_score_across_extracted_frames) per file.

    Per-file rather than per-frame because a clip is the atomic unit for the
    alert path: we vote across frames of a clip. Distributions over per-clip
    max are what actually govern alert-threshold behaviour.
    """
    results = []
    for i, path in enumerate(files, start=1):
        scores = [score_frame(detector, f) for f in iter_frames(path, step, transform)]
        if not scores:
            print(f"  [skip] {path.name}: no frames", file=sys.stderr)
            continue
        best = max(scores)
        results.append((path, best))
        if i % 5 == 0 or i == len(files):
            print(f"  {label}: {i}/{len(files)}  {path.name} best={best:.3f}",
                  file=sys.stderr)
    return results


def frame_scores(files: List[Path], detector: ObjectDetector,
                 step: int, label: str,
                 transform: Optional[Transform] = None) -> List[float]:
    """Per-FRAME scores across every file. Needed for deter-threshold policy,
    which fires on a single frame."""
    all_scores = []
    for i, path in enumerate(files, start=1):
        for f in iter_frames(path, step, transform):
            all_scores.append(score_frame(detector, f))
        if i % 5 == 0 or i == len(files):
            print(f"  {label}: {i}/{len(files)}", file=sys.stderr)
    return all_scores


def per_clip_scores(files: List[Path], detector: ObjectDetector,
                    step: int,
                    transform: Optional[Transform] = None) -> List[List[float]]:
    """Per-clip lists of per-frame scores. Feeds voting analysis."""
    out = []
    for path in files:
        scores = [score_frame(detector, f) for f in iter_frames(path, step, transform)]
        if scores:
            out.append(scores)
    return out


def voting_recall(per_clip_frame_scores: List[List[float]],
                  threshold: float, k: int) -> float:
    """Fraction of clips where at least k frames score >= threshold."""
    if not per_clip_frame_scores:
        return 0.0
    hits = sum(1 for scores in per_clip_frame_scores
               if sum(s >= threshold for s in scores) >= k)
    return hits / len(per_clip_frame_scores)


def voting_precision_proxy(negative_per_clip_frame_scores: List[List[float]],
                           threshold: float, k: int) -> float:
    """Fraction of NEGATIVE clips voting NEGATIVE. 1.0 = perfect."""
    if not negative_per_clip_frame_scores:
        return 1.0
    passes = sum(1 for scores in negative_per_clip_frame_scores
                 if sum(s >= threshold for s in scores) >= k)
    return 1.0 - passes / len(negative_per_clip_frame_scores)


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return float('nan')
    return float(np.percentile(vals, p))


def collect_paths(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            out.append(p)
    return out


def describe(name: str, vals: List[float]) -> None:
    print(f"\n{name}: n={len(vals)}")
    if not vals:
        print("  (no data)")
        return
    for p in (50, 90, 95, 99):
        print(f"  p{p:>2}: {percentile(vals, p):.3f}")
    print(f"  max: {max(vals):.3f}   mean: {sum(vals) / len(vals):.3f}")


# Defaults are separated so tests and the guard can reference them by name.
# See the header for the ground-truth principle these encode.
DEFAULT_POSITIVES = Path('../heron-data/heron_project/dataset/val/images')
DEFAULT_NEGATIVES = Path('media')


class OverlappingSetsError(RuntimeError):
    """Positives and negatives share at least one file, or one directory is
    nested under the other. The two sets must be disjoint or the measurement
    is measuring itself."""


def check_disjoint(pos_files: List[Path], neg_files: List[Path],
                   pos_root: Path, neg_root: Path) -> None:
    """Refuse to run if positives and negatives overlap.

    Two cases both count as overlap:
      1. A file appears in both lists (same resolved path).
      2. One root is inside the other. Even if no files overlap today, this
         is the exact shape of the old bug where --positives=media/heron and
         --negatives=media silently subtracted one from the other. Fail before
         the numbers get published.

    An empty positives set is a separate error (nothing to measure), reported
    upstream with more context — not conflated with overlap here.
    """
    # Structural checks first — the resulting error message points at the
    # config problem (--positives vs --negatives) rather than the resulting
    # file collision, which is downstream.
    pos_root_r = pos_root.resolve()
    neg_root_r = neg_root.resolve()
    if pos_root_r == neg_root_r:
        raise OverlappingSetsError(
            f"--positives and --negatives point at the same directory "
            f"({pos_root_r}). One set is entirely inside the other."
        )
    if pos_root_r in neg_root_r.parents:
        raise OverlappingSetsError(
            f"--negatives ({neg_root_r}) is nested under --positives "
            f"({pos_root_r}). One set is entirely inside the other; the "
            f"previous silent-subtraction bug lived in exactly this shape."
        )
    if neg_root_r in pos_root_r.parents:
        raise OverlappingSetsError(
            f"--positives ({pos_root_r}) is nested under --negatives "
            f"({neg_root_r}). One set is entirely inside the other; the "
            f"previous silent-subtraction bug lived in exactly this shape."
        )

    # Structurally disjoint, but a specific file may still be listed twice
    # (symlink, hard link, duplicate placement). Catch that too.
    pos_resolved = {p.resolve() for p in pos_files}
    neg_resolved = {p.resolve() for p in neg_files}
    shared = pos_resolved & neg_resolved
    if shared:
        sample = sorted(str(p) for p in list(shared)[:3])
        raise OverlappingSetsError(
            f"positives and negatives share {len(shared)} file(s), e.g.:\n  "
            + "\n  ".join(sample)
            + "\nGround truth requires disjoint sets. Point --positives and "
              "--negatives at unrelated directories."
        )


def _pick_alert_threshold(pos_per_clip: List[List[float]],
                          neg_per_clip: List[List[float]],
                          k: int) -> Tuple[float, float, float]:
    """Sweep candidate alert thresholds and return the (t, recall, precision)
    that maximises recall subject to precision-proxy >= 0.99. Falls back to
    the highest-precision candidate if nothing meets that bar."""
    if not (pos_per_clip and neg_per_clip):
        return 0.95, 0.0, 1.0
    best = None
    for t in np.arange(0.60, 0.981, 0.01):
        t = float(round(t, 2))
        prec = voting_precision_proxy(neg_per_clip, t, k)
        rec  = voting_recall(pos_per_clip, t, k)
        if prec >= 0.99 and (best is None or rec > best[1]):
            best = (t, rec, prec)
    if best is None:
        for t in np.arange(0.60, 0.981, 0.01):
            t = float(round(t, 2))
            prec = voting_precision_proxy(neg_per_clip, t, k)
            rec  = voting_recall(pos_per_clip, t, k)
            if best is None or prec > best[2] or (prec == best[2] and rec > best[1]):
                best = (t, rec, prec)
    return best


def _build_transform(source: str, sim_mode: str,
                     target_wh: Tuple[int, int]) -> Optional[Transform]:
    """Return the frame transform to apply to positives before scoring.

    source='camera'  -> None (pass through; frames already came from a camera)
    source='stock'   -> simulate_camera(..., mode=sim_mode)
    """
    if source == 'camera':
        return None
    return lambda img: simulate_camera(img, target_wh, mode=sim_mode)


def _measure_positives(files: List[Path], detector: ObjectDetector,
                       step: int, label: str,
                       transform: Optional[Transform]
                       ) -> Tuple[List[float], List[List[float]]]:
    """Score positives once with a single transform. Returns (per_frame,
    per_clip). Runs frame_scores and per_clip_scores in one pass to avoid
    double inference — every frame is inferred once, not twice."""
    all_scores: List[float] = []
    per_clip: List[List[float]] = []
    for i, path in enumerate(files, start=1):
        clip: List[float] = []
        for f in iter_frames(path, step, transform):
            s = score_frame(detector, f)
            clip.append(s)
            all_scores.append(s)
        if clip:
            per_clip.append(clip)
        if i % 5 == 0 or i == len(files):
            print(f"  {label}: {i}/{len(files)}", file=sys.stderr)
    return all_scores, per_clip


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--positives', type=Path, default=DEFAULT_POSITIVES,
                    help=f"dir of images or MP4s containing herons "
                         f"(default: {DEFAULT_POSITIVES} — 44 stock heron "
                         f"images, provisional until field positives exist)")
    ap.add_argument('--negatives', type=Path, default=DEFAULT_NEGATIVES,
                    help=f"dir of images or MP4s containing NO herons "
                         f"(default: {DEFAULT_NEGATIVES}, walked recursively; "
                         f"every clip in there is a verified true negative)")
    ap.add_argument('--positives-source', choices=('stock', 'camera'),
                    default='stock',
                    help="Provenance of the positives directory. 'stock' = "
                         "high-resolution photos, apply camera simulation "
                         "before scoring so measurement matches deployment. "
                         "'camera' = frames already came from a webcam at "
                         "system.resolution — pass through untouched. Getting "
                         "this wrong once pond footage arrives would waste "
                         "the data (double-processing camera-native frames).")
    ap.add_argument('--sim-mode', choices=('crop', 'pad', 'both'),
                    default='both',
                    help="Camera-simulation policy applied to stock positives. "
                         "'crop' (canonical): centre-crop to camera aspect, "
                         "downsample. 'pad': fit-and-pad with grey. 'both': "
                         "run positives twice and print a sensitivity table. "
                         "Ignored when --positives-source=camera.")
    ap.add_argument('--step', type=int, default=DEFAULT_STEP,
                    help=f"take every Nth frame from each clip (default {DEFAULT_STEP})")
    ap.add_argument('--k', type=int, default=3,
                    help="N-of-M voting: min hits (default 3)")
    ap.add_argument('--m', type=int, default=5,
                    help="N-of-M voting: window size (default 5, matches key-frame count)")
    args = ap.parse_args()

    if not args.positives.exists():
        print(f"ERROR: positives directory not found: {args.positives}\n"
              f"  Point --positives at a curated set of heron images or clips.\n"
              f"  Default expects ../heron-data/heron_project/dataset/val/images/\n"
              f"  next to this repo. Clone or symlink heron-data alongside "
              f"heron_deterrent, or pass --positives explicitly.",
              file=sys.stderr)
        return 2

    if not args.negatives.exists():
        print(f"ERROR: negatives directory not found: {args.negatives}",
              file=sys.stderr)
        return 2

    config = Config('config.yaml')
    loader = ModelLoader(config.get('ai.target', 'cpu'),
                         config.get('ai.num_threads', 4))
    labels = {int(k): v for k, v in (config.get('ai.labels', {}) or {}).items()}
    detector = ObjectDetector(loader, confidence_threshold=SCORE_FLOOR,
                              labels=labels or None)

    target_wh = (config.get('system.resolution.width', 640),
                 config.get('system.resolution.height', 480))
    print(f"Camera target for simulation: {target_wh[0]}x{target_wh[1]}",
          file=sys.stderr)

    # ---- Collect file lists and refuse overlap -----------------------------
    pos_files = collect_paths(args.positives)
    neg_files = collect_paths(args.negatives)
    try:
        check_disjoint(pos_files, neg_files, args.positives, args.negatives)
    except OverlappingSetsError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2

    # ---- Negatives ---------------------------------------------------------
    # No simulation on negatives: they are already camera-native and their
    # distribution matches an independent 275-frame measurement to three
    # decimal places. Do not touch this path.
    print(f"\n=== Negatives ({args.negatives}) ===", file=sys.stderr)
    print(f"  {len(neg_files)} file(s)", file=sys.stderr)
    neg_frame_scores = frame_scores(neg_files, detector, args.step, 'neg')
    neg_per_clip = per_clip_scores(neg_files, detector, args.step)

    # ---- Positives ---------------------------------------------------------
    # Which simulation modes to run positives through. When source=camera,
    # only the identity transform makes sense (frames already came from one).
    if args.positives_source == 'camera':
        modes = ['identity']
        transforms = {'identity': None}
    elif args.sim_mode == 'both':
        modes = ['crop', 'pad']
        transforms = {m: _build_transform('stock', m, target_wh) for m in modes}
    else:
        modes = [args.sim_mode]
        transforms = {args.sim_mode: _build_transform('stock', args.sim_mode, target_wh)}

    print(f"\n=== Positives ({args.positives}) "
          f"source={args.positives_source} modes={modes} ===", file=sys.stderr)
    print(f"  {len(pos_files)} file(s)", file=sys.stderr)

    pos_frame_by_mode: dict = {}
    pos_per_clip_by_mode: dict = {}
    for mode in modes:
        print(f"  scoring positives under sim={mode}...", file=sys.stderr)
        frames, per_clip = _measure_positives(
            pos_files, detector, args.step, f'pos({mode})',
            transforms[mode],
        )
        pos_frame_by_mode[mode] = frames
        pos_per_clip_by_mode[mode] = per_clip

    # Canonical mode: what the "Suggested config" line uses. Crop matches a
    # real camera; identity is the only option for camera-source runs; pad is
    # explicitly not policy.
    if 'crop' in modes:
        canonical = 'crop'
    elif 'identity' in modes:
        canonical = 'identity'
    else:
        canonical = modes[0]

    # ---- Report ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Negatives (unchanged, no simulation applied)")
    describe('per-frame', neg_frame_scores)

    print("\n" + "=" * 60)
    print(f"Positives sensitivity (canonical={canonical})")
    if len(modes) > 1:
        _sensitivity_table(pos_frame_by_mode)
    else:
        describe(f'positives ({canonical}, per-frame)',
                 pos_frame_by_mode[canonical])

    # ---- Deter threshold: govern per-frame FPs (canonical positives) -----
    canonical_pos_frames = pos_frame_by_mode[canonical]
    canonical_pos_per_clip = pos_per_clip_by_mode[canonical]

    if neg_frame_scores:
        deter = min(1.0, round(percentile(neg_frame_scores, 99) + 0.01, 3))
        deter_recall = (sum(s >= deter for s in canonical_pos_frames)
                        / max(len(canonical_pos_frames), 1))
        deter_fpr = sum(s >= deter for s in neg_frame_scores) / len(neg_frame_scores)
    else:
        deter, deter_recall, deter_fpr = 0.90, 0.0, 0.0

    alert_t, alert_recall, alert_prec = _pick_alert_threshold(
        canonical_pos_per_clip, neg_per_clip, args.k)

    print("\n" + "=" * 60)
    print(f"Suggested config (measured, canonical positives={canonical})")
    print(f"\n  ai.deter_threshold: {deter}")
    print(f"    per-frame:  recall={deter_recall:.2f}  fpr={deter_fpr:.3f}")
    print(f"    intent:     one confident frame fires the sound; false bang cheap, "
          f"late bang expensive.")
    print(f"\n  ai.alert_threshold: {alert_t}")
    print(f"  ai.alert_vote_k:    {args.k}")
    print(f"  ai.alert_vote_m:    {args.m}")
    print(f"    per-clip N-of-M ({args.k}-of-{args.m}):"
          f" recall={alert_recall:.2f}  precision-proxy={alert_prec:.2f}")
    print(f"    intent:     record + notify only when the clip sustains a heron "
          f"call across frames.")

    # ---- What the CURRENT config would score ------------------------------
    current_deter = config.get('ai.deter_threshold',
                               config.get('ai.confidence_threshold', 0.90))
    current_alert = config.get('ai.alert_threshold',
                               config.get('ai.confidence_threshold', 0.95))
    current_k = config.get('ai.alert_vote_k', args.k)
    if neg_frame_scores:
        cur_deter_fpr = sum(s >= current_deter for s in neg_frame_scores) / len(neg_frame_scores)
    else:
        cur_deter_fpr = float('nan')

    print("\n" + "-" * 60)
    print("Current config for comparison")
    print(f"  ai.deter_threshold={current_deter}, fpr={cur_deter_fpr:.3f}")
    # Show recall under every mode so the sensitivity is visible at the
    # decision line, not buried in the sensitivity table.
    for mode in modes:
        frames = pos_frame_by_mode[mode]
        rec = sum(s >= current_deter for s in frames) / max(len(frames), 1)
        marker = ' (canonical)' if mode == canonical else ''
        print(f"    recall under sim={mode}{marker}: {rec:.2f}")
    print(f"  ai.alert_threshold={current_alert} @ {current_k}-of-{args.m}:")
    for mode in modes:
        per_clip = pos_per_clip_by_mode[mode]
        rec = voting_recall(per_clip, current_alert, current_k)
        prec = voting_precision_proxy(neg_per_clip, current_alert, current_k)
        marker = ' (canonical)' if mode == canonical else ''
        print(f"    sim={mode}{marker}: recall={rec:.2f}  precision-proxy={prec:.2f}")

    print(f"\nCopy into config.yaml -> ai:")
    print(f"  deter_threshold: {deter}")
    print(f"  alert_threshold: {alert_t}")
    print(f"  alert_vote_k: {args.k}")
    print(f"  alert_vote_m: {args.m}")

    if args.positives == DEFAULT_POSITIVES:
        print("\nNOTE: --positives is the val split of the stock training dataset. "
              "Same domain as training, so recall above flatters the model even "
              "with camera simulation applied. Point --positives at a curated "
              "field-labelled set once the dashboard has produced one; use "
              "--positives-source camera when you do.")


def _sensitivity_table(pos_frame_by_mode: dict) -> None:
    """Side-by-side per-frame percentiles under each simulation mode."""
    modes = list(pos_frame_by_mode.keys())
    header = f"  {'stat':>10}  " + "  ".join(f"{m:>8}" for m in modes)
    print(header)
    for label, stat in (('n', len),
                        ('median', lambda v: percentile(v, 50)),
                        ('p90', lambda v: percentile(v, 90)),
                        ('p95', lambda v: percentile(v, 95)),
                        ('p99', lambda v: percentile(v, 99)),
                        ('max', max),
                        ('mean', lambda v: sum(v) / len(v))):
        row = f"  {label:>10}  " + "  ".join(
            f"{stat(pos_frame_by_mode[m]):>8.3f}"
            if label != 'n' else f"{len(pos_frame_by_mode[m]):>8d}"
            for m in modes
        )
        print(row)


if __name__ == '__main__':
    sys.exit(main() or 0)
