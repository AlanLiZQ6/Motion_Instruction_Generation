"""
mirror_handedness.py
====================
One-time preprocessing: mirror all data for left-handed players so that
every downstream algorithm (DTW, labeling, training) sees consistent
right-handed motion.

Three things are mirrored in-place:
  1. data/raw_data  — .avi videos (horizontal flip)
  2. data/skeletons        — MediaPipe (T, 33, 4) .npy files
  3. data/skeletons_smpl   — SMPL (T, 22, 3) .npy files

Usage:
    # Dry run — prints what would change, touches nothing
    py mirror_handedness.py --dry-run

    # Real run
    py mirror_handedness.py
"""

import argparse
import os
import sys
import numpy as np
import cv2

# ──────────────────────────────────────────────
# Confirmed left-handed player IDs
# ──────────────────────────────────────────────
LEFTY_IDS = {"p5", "p7", "p8", "p19", "p24", "p46", "p48", "p52"}

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
VIDEO_DIR    = os.path.join(BASE, "raw_data")
MEDIAPIPE_DIR = os.path.join(BASE, "skeletons")
SMPL_DIR      = os.path.join(BASE, "skeletons_smpl")


# ──────────────────────────────────────────────
# MediaPipe 33-joint left↔right pairs
# ──────────────────────────────────────────────
MEDIAPIPE_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6),          # eye inner/centre/outer
    (7, 8),                           # ears
    (9, 10),                          # mouth corners
    (11, 12), (13, 14), (15, 16),    # shoulder / elbow / wrist
    (17, 18), (19, 20), (21, 22),    # pinky / index / thumb
    (23, 24), (25, 26), (27, 28),    # hip / knee / ankle
    (29, 30), (31, 32),              # heel / foot index
]

# SMPL 22-joint left↔right pairs
SMPL_LR_PAIRS = [
    (1, 2), (4, 5), (7, 8), (10, 11),
    (13, 14), (16, 17), (18, 19), (20, 21),
]


def is_lefty(filename: str) -> bool:
    pid = filename.split("_")[0].lower()
    return pid in LEFTY_IDS


# ──────────────────────────────────────────────
# Mirror helpers
# ──────────────────────────────────────────────
def mirror_npy(path: str, lr_pairs: list, dry_run: bool):
    data = np.load(path)
    out = data.copy()
    out[..., 0] *= -1                              # flip X
    for l, r in lr_pairs:
        out[..., [l, r], :] = out[..., [r, l], :] # swap left ↔ right joints
    if not dry_run:
        np.save(path, out)


def mirror_video(path: str, dry_run: bool):
    if dry_run:
        return

    cap = cv2.VideoCapture(path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    tmp = path + ".tmp.avi"
    out = cv2.VideoWriter(tmp, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(cv2.flip(frame, 1))
    cap.release()
    out.release()

    os.replace(tmp, path)


# ──────────────────────────────────────────────
# Walk a directory tree and process matching files
# ──────────────────────────────────────────────
def process_dir(root: str, ext: str, mirror_fn, label: str, dry_run: bool):
    total = processed = 0
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.endswith(ext):
                continue
            if not is_lefty(fname):
                continue
            total += 1
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, BASE)
            if dry_run:
                print(f"  [DRY-RUN] would mirror {label}: {rel}")
            else:
                print(f"  Mirroring {label}: {rel} ...", end=" ", flush=True)
                mirror_fn(fpath)
                print("done")
            processed += 1
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be changed without modifying any files")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN — no files will be modified ===\n")
    else:
        print("Left-handed player IDs:", sorted(LEFTY_IDS))
        print("Directories:")
        print(f"  videos    : {VIDEO_DIR}")
        print(f"  mediapipe : {MEDIAPIPE_DIR}")
        print(f"  smpl      : {SMPL_DIR}")
        answer = input("\nThis will overwrite files in-place. Continue? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)
        print()

    n_vid = process_dir(
        VIDEO_DIR, ".avi",
        lambda p: mirror_video(p, dry_run=False),
        "video", args.dry_run,
    )

    # UNCOMMENT IF PROCESSING SKELETONS
    # n_mp = process_dir(
    #     MEDIAPIPE_DIR, ".npy",
    #     lambda p: mirror_npy(p, MEDIAPIPE_LR_PAIRS, dry_run=False),
    #     "mediapipe", args.dry_run,
    # )

    # n_smpl = process_dir(
    #     SMPL_DIR, ".npy",
    #     lambda p: mirror_npy(p, SMPL_LR_PAIRS, dry_run=False),
    #     "smpl", args.dry_run,
    # )

    print(f"\n{'[DRY RUN] Would process' if args.dry_run else 'Processed'}:")
    print(f"  {n_vid:3d} videos")
    # print(f"  {n_mp:3d} mediapipe skeletons")
    # print(f"  {n_smpl:3d} smpl skeletons")

    if not args.dry_run:
        print("\nAll done. Next step: remove the on-the-fly flip logic from")
        print("  dataset_construct/data_labeling_revamped.py  (lines ~172-179)")


if __name__ == "__main__":
    main()
