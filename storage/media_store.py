# ============================================================
# FILE: storage/media_store.py
# ============================================================

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import logging

from capture.frame_classifier import FRAME_TYPES, UNSETTLED
from storage.video_encoder import encode_h264

logger = logging.getLogger(__name__)

# Ephemeral clip storage: subject to age retention and a hard size cap.
# Deleting the oldest clip is fine — a new one will show up on the next event.
#
# Two pools, each with its OWN cap, so neither can evict the other:
#   classified — clips a model looked at (heron/, other/, unknown/).
#   motion     — clips no model looked at: night motion-only triggers and
#                anything captured while the frame type was unsettled.
# A noisy night (moths in front of the IR LEDs, rain on the housing) can fill
# the motion pool to its cap and start evicting its own oldest clips, but it
# can never push out an unlabelled daylight clip.
CLIP_POOLS = {
    'classified': ('heron', 'other', 'unknown'),
    'motion': ('motion',),
}
EPHEMERAL_MEDIA_SUBDIRS = tuple(d for pool in CLIP_POOLS.values() for d in pool)

# Curated training data: NEVER deleted by retention or the size cap.
# Two failure modes make automated deletion under labeled/ actively poisonous:
#   1. The size cap sorts by mtime and stops as soon as it is under the ceiling,
#      so it deletes the 100 KB .jpg first and leaves the 50 B .txt orphaned.
#   2. Worse in the other direction: if a .jpg outlives its label, YOLO sees the
#      image with no annotation and treats it as a background (negative) example.
#      A hand-labelled heron silently becomes "no heron here" — the exact
#      poisoning the manual labelling exists to prevent.
# The labeled/ directory has its own soft warning (labeled_bytes + max_labeled_gb)
# so growth is visible without ever being unsafe.
#
# Labelled data is split by image domain, one YOLO tree per frame type:
#     labeled/day/{images,labels}   labeled/ir/{images,labels}   labeled/dark/...
# A training run pointed at labeled/day physically cannot see a night frame.
# Mixing them would teach a combined model "monochrome means no heron", because
# IR negatives are plentiful and IR positives rare.
#
# labeled/images and labeled/labels are the pre-split flat layout. Nothing
# writes there any more; they stay protected in case an older device has data
# in them, and preflight warns if they are non-empty.
LABELED_ROOT = 'labeled'
LEGACY_LABELED_SUBDIRS = ('labeled/images', 'labeled/labels')
CURATED_MEDIA_SUBDIRS = tuple(
    f'{LABELED_ROOT}/{ft}/{kind}' for ft in FRAME_TYPES for kind in ('images', 'labels')
) + LEGACY_LABELED_SUBDIRS

assert not (set(EPHEMERAL_MEDIA_SUBDIRS) & set(CURATED_MEDIA_SUBDIRS)), (
    "A subdir listed in both sets would be deletable — labeled/ must not appear "
    "in the ephemeral tuple."
)
assert not any(d == LABELED_ROOT or d.startswith(LABELED_ROOT + '/')
               for d in EPHEMERAL_MEDIA_SUBDIRS), (
    "No clip pool may live under labeled/ — the retention sweep and size cap "
    "would then reach training data."
)


class MediaStore:
    def __init__(self, base_path: str, fps: float = 5.0):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        # Write clips at the rate they were captured, so they play at real speed.
        self.fps = fps

        # Create subdirectories
        for subdir in EPHEMERAL_MEDIA_SUBDIRS:
            (self.base_path / subdir).mkdir(exist_ok=True)
        for subdir in CURATED_MEDIA_SUBDIRS:
            if subdir not in LEGACY_LABELED_SUBDIRS:
                (self.base_path / subdir).mkdir(parents=True, exist_ok=True)

    def save_video_clip(self, frames: List[np.ndarray], label: str,
                        frame_type: str) -> str:
        """Write a clip. The frame type goes into the filename as well as the
        database row, so a clip copied off the card still says what it is:
        heron_day_<ts>.mp4, motion_ir_<ts>.mp4, motion_switching_<ts>.mp4.
        """
        if frame_type not in FRAME_TYPES + UNSETTLED:
            raise ValueError(f"unknown frame_type {frame_type!r}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{label}_{frame_type}_{timestamp}.mp4"

        # Determine subdirectory
        if label in ("heron", "unknown", "motion"):
            subdir = label
        else:
            subdir = "other"
        
        filepath = self.base_path / subdir / filename

        try:
            if len(frames) == 0:
                logger.warning("No frames to save")
                return str(filepath)

            # H.264 via ffmpeg, NOT cv2.VideoWriter(*'mp4v'). The old writer
            # produced MPEG-4 Part 2, which no browser can decode: the gallery
            # drew working playback controls over a permanently black frame and
            # reported no error. See storage/video_encoder.py for why cv2 cannot
            # do H.264 on this hardware and what it costs to do it properly.
            encode_h264(frames, str(filepath), self.fps)
            logger.info(f"Saved video: {filepath}")

            return str(filepath)

        except Exception as e:
            # Loud, and with no unplayable file left behind — encode_h264
            # removes its own partial output. Returning "" keeps a single failed
            # clip from killing the detection thread, but the error is never
            # swallowed into a silently-broken file.
            logger.error(f"Failed to save video {filepath}: {e}")
            return ""
    
    def save_labeled_image(self, image: np.ndarray, label: str,
                          bboxes: List[Tuple], detection_id: int,
                          frame_type: str) -> Tuple[str, str]:
        # DISAMBIGUATING OLD EMPTY .txt FILES
        # -----------------------------------
        # An empty .txt under labeled/labels/ has two possible provenances:
        #
        #   * INTENDED: label was 'cat'/'dog'/'bird'/'person'/'unknown'. Empty
        #     annotation is the correct YOLO convention for a background
        #     (negative) example. Keep it.
        #   * BUG:      label was 'heron' and the UI sent bboxes=[] because the
        #     old dashboard hard-coded that. Trains the model that the frame
        #     contains NO heron. Delete or re-label.
        #
        # They are tellable apart from the filename: base_name is
        # 'labeled_{detection_id}_{timestamp}', so the detection ID is in the
        # path. Join to the detections table on that id:
        #
        #   sqlite> SELECT id, user_label FROM detections WHERE id IN (
        #             SELECT DISTINCT CAST(substr(basename, 9,
        #                     instr(substr(basename, 9), '_') - 1) AS INTEGER)
        #             FROM (<empty .txt basenames>)
        #           );
        #
        # user_label = 'heron' + empty .txt = bug. Anything else + empty .txt =
        # correct behaviour. No repair script ships here because labeled/ is
        # currently empty — nothing was ever harmed. If an older device turns
        # up with a populated labeled/, use the join above to disambiguate.
        #
        # DOMAIN SEPARATION
        # -----------------
        # frame_type picks the tree: labeled/<frame_type>/{images,labels}. It
        # must be a settled type — an establishing/switching frame has no
        # domain, and guessing one is how day and night data get mixed. The
        # caller decides the domain (see ui/app.py resolve_label_domain); this
        # method refuses anything it cannot place.
        if frame_type not in FRAME_TYPES:
            raise ValueError(
                f"cannot file a labelled image under frame_type {frame_type!r}; "
                f"must be one of {', '.join(FRAME_TYPES)}"
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"labeled_{detection_id}_{timestamp}"
        domain_root = self.base_path / LABELED_ROOT / frame_type

        # Save image
        image_dir = domain_root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{base_name}.jpg"
        cv2.imwrite(str(image_path), image)

        # Save YOLO format label
        label_dir = domain_root / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / f"{base_name}.txt"
        
        height, width = image.shape[:2]
        
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                x, y, w, h = bbox
                # Convert to YOLO format (normalized center coordinates)
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                norm_w = w / width
                norm_h = h / height
                
                # The model is single-class, so only heron boxes are valid
                # annotations. Anything else is written as a background image
                # (an empty label file), which is what YOLO expects for a
                # negative example — never as a fictitious class 1.
                if label == "heron":
                    f.write(f"0 {x_center} {y_center} {norm_w} {norm_h}\n")

        logger.info(f"Saved labeled data: {frame_type}/{base_name}"
                    + ("" if label == "heron" else " (as a background/negative example)"))
        return str(image_path), str(label_path)

    def _files_under(self, subdirs) -> List[Path]:
        files = []
        for subdir in subdirs:
            path = self.base_path / subdir
            if not path.exists():
                continue
            files.extend(f for f in path.iterdir() if f.is_file())
        return files

    def _ephemeral_files(self) -> List[Path]:
        """Files that retention and the size cap may delete.

        Explicitly excludes curated training data — see CURATED_MEDIA_SUBDIRS.
        """
        return self._files_under(EPHEMERAL_MEDIA_SUBDIRS)

    def _curated_files(self) -> List[Path]:
        """Every file anywhere under labeled/, recursively.

        Recursive on purpose: _files_under() only looks one level down, so a
        new subdirectory that was not added to a list would silently fall out
        of the size accounting. Anything under labeled/ is curated, whether
        or not a list here knows its name.
        """
        root = self.base_path / LABELED_ROOT
        if not root.exists():
            return []
        return [f for f in root.rglob('*') if f.is_file()]

    def legacy_labeled_files(self) -> List[Path]:
        """Files in the pre-split flat labeled/images|labels layout. These have
        no recorded domain; a human needs to sort them into labeled/<type>/."""
        return self._files_under(LEGACY_LABELED_SUBDIRS)

    def _bytes_of(self, files) -> int:
        total = 0
        for f in files:
            try:
                total += f.stat().st_size
            except OSError:
                pass  # deleted underneath us; ignore
        return total

    def total_bytes(self) -> int:
        """Bytes used across ALL media, ephemeral and curated.

        The low-disk warning is about the partition as a whole, so it counts
        labeled/ even though the cleanup paths refuse to touch it.
        """
        return self._bytes_of(self._ephemeral_files()) + self._bytes_of(self._curated_files())

    def labeled_bytes(self) -> int:
        return self._bytes_of(self._curated_files())

    def cleanup_old_media(self, days: int) -> int:
        """Delete ephemeral media older than `days`.

        labeled/ is deliberately excluded — see CURATED_MEDIA_SUBDIRS for why
        automated deletion of hand-labelled training data is unsafe.
        """
        cutoff = datetime.now().timestamp() - (days * 86400)
        removed = 0
        for f in self._ephemeral_files():
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                pass
        if removed:
            logger.info(f"Retention: deleted {removed} file(s) older than {days} days")
        return removed

    def enforce_size_cap(self, max_gb: float, pool: str = 'classified') -> int:
        """Delete oldest clips in `pool` until that pool is under `max_gb`.

        Retention alone cannot protect the card: at 100 events/day a 2 GB
        partition fills in about four months, long before a 365-day cutoff
        removes anything. This is the backstop that actually bounds growth.

        Each pool is capped separately — see CLIP_POOLS. labeled/ is in no
        pool; use warn_labeled_size() for a visible-but-non-destructive check.
        """
        max_bytes = int(max_gb * (1024 ** 3))
        files = []
        for f in self._files_under(CLIP_POOLS[pool]):
            try:
                files.append((f.stat().st_mtime, f.stat().st_size, f))
            except OSError:
                pass

        total = sum(size for _, size, _ in files)
        if total <= max_bytes:
            return 0

        files.sort(key=lambda item: item[0])  # oldest first
        removed = 0
        for _, size, f in files:
            if total <= max_bytes:
                break
            try:
                f.unlink()
                total -= size
                removed += 1
            except OSError:
                pass

        logger.warning(
            f"Size cap: deleted {removed} oldest file(s) to bring the {pool} "
            f"pool under {max_gb:.1f} GB (now {total / 1024 ** 3:.2f} GB)"
        )
        return removed

    def warn_labeled_size(self, max_labeled_gb: float) -> bool:
        """Warn if the curated training data is over `max_labeled_gb`.

        Returns True if a warning was emitted. Never deletes anything: growth
        under labeled/ is a signal to move data off the card, not to lose it.
        """
        used_bytes = self.labeled_bytes()
        max_bytes = int(max_labeled_gb * (1024 ** 3))
        if used_bytes <= max_bytes:
            return False
        logger.warning(
            f"labeled/ is using {used_bytes / 1024 ** 3:.2f} GB, over the "
            f"{max_labeled_gb:.1f} GB soft limit. Nothing will be deleted — "
            f"move it off the card: "
            f"rsync -a {self.base_path}/labeled/ <backup>:/labeled/"
        )
        return True


    def get_file_path(self, filename: str) -> Optional[Path]:
        image_dirs = [d for d in CURATED_MEDIA_SUBDIRS if d.endswith('/images')]
        for subdir in list(EPHEMERAL_MEDIA_SUBDIRS) + image_dirs:
            filepath = self.base_path / subdir / filename
            if filepath.exists():
                return filepath
        return None