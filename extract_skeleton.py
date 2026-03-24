"""
Skeleton Extraction + Visualization from Video using MediaPipe (v0.10.20+)
==========================================================================
Setup:
    pip install mediapipe opencv-python numpy

    # Download the pose model (run once):
    wget -O pose_landmarker_heavy.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

Usage:
    # Single video — saves .npy + .mp4
    python extract_skeleton.py --single /path/to/video.avi

    # Batch process entire THETIS dataset
    python extract_skeleton.py --input_dir /path/to/THETIS/Video_RGB --output_dir ./skeleton_output

Output per video:
    - {name}.npy            (T, 33, 5) — normalized x, y, z, visibility, presence
    - {name}_world.npy      (T, 33, 3) — 3D world coords in meters (hip-centered)
    - {name}_skeleton.mp4   — annotated video with skeleton overlay
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
import json
import time

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ──────────────────────────────────────────────
# MediaPipe pose connections (for drawing bones)
# ──────────────────────────────────────────────
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32),
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]

# Color per body part (BGR for OpenCV)
BONE_COLORS = {
    'torso':     (255, 165, 0),    # orange
    'left_arm':  (0, 255, 0),      # green
    'right_arm': (0, 200, 255),    # yellow
    'left_leg':  (255, 0, 255),    # magenta
    'right_leg': (0, 0, 255),      # red
    'face':      (200, 200, 200),  # gray
}

LEFT_ARM_JOINTS  = {11, 13, 15, 17, 19, 21}
RIGHT_ARM_JOINTS = {12, 14, 16, 18, 20, 22}
LEFT_LEG_JOINTS  = {23, 25, 27, 29, 31}
RIGHT_LEG_JOINTS = {24, 26, 28, 30, 32}
TORSO_PAIRS = {(11,12), (11,23), (12,24), (23,24)}


def get_bone_color_bgr(i, j):
    pair = (min(i,j), max(i,j))
    if pair in TORSO_PAIRS:
        return BONE_COLORS['torso']
    if i in LEFT_ARM_JOINTS or j in LEFT_ARM_JOINTS:
        return BONE_COLORS['left_arm']
    if i in RIGHT_ARM_JOINTS or j in RIGHT_ARM_JOINTS:
        return BONE_COLORS['right_arm']
    if i in LEFT_LEG_JOINTS or j in LEFT_LEG_JOINTS:
        return BONE_COLORS['left_leg']
    if i in RIGHT_LEG_JOINTS or j in RIGHT_LEG_JOINTS:
        return BONE_COLORS['right_leg']
    return BONE_COLORS['face']


def draw_skeleton_on_frame(frame, landmarks_norm, w, h):
    """Draw skeleton overlay on a BGR frame. landmarks_norm: list of 33 landmarks."""
    pts = []
    for lm in landmarks_norm:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))

    # Draw bones
    for (i, j) in POSE_CONNECTIONS:
        color = get_bone_color_bgr(i, j)
        cv2.line(frame, pts[i], pts[j], color, 2, cv2.LINE_AA)

    # Draw joints
    for idx, (x, y) in enumerate(pts):
        if idx <= 10:
            # Face joints — smaller
            cv2.circle(frame, (x, y), 2, (200, 200, 200), -1, cv2.LINE_AA)
        else:
            # Body joints
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────
# Model path helper
# ──────────────────────────────────────────────
MODEL_URLS = {
    'heavy': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
    'full':  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
    'lite':  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
}


def get_model_path(model_type: str = 'heavy') -> str:
    filename = f'pose_landmarker_{model_type}.task'
    for search_dir in ['.', os.path.dirname(os.path.abspath(__file__)), os.path.expanduser('~/.mediapipe')]:
        path = os.path.join(search_dir, filename)
        if os.path.exists(path):
            return path
    url = MODEL_URLS.get(model_type, MODEL_URLS['heavy'])
    print(f"\nModel file '{filename}' not found!")
    print(f"Download it with:\n  wget -O {filename} {url}")
    raise FileNotFoundError(f"Model file '{filename}' not found. Download from: {url}")


# ──────────────────────────────────────────────
# Core: extract + visualize in one pass
# ──────────────────────────────────────────────
def process_video(
    video_path: str,
    model_path: str,
    output_npy_path: str,
    output_world_npy_path: str,
    output_video_path: str,
    num_poses: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    save_npy: bool = True,
    save_world: bool = True,
    save_video: bool = True,
) -> dict:
    """
    Extract skeleton data AND render annotated video in a single pass.

    Saves:
        output_npy_path:        (T, 33, 5) normalized landmarks
        output_world_npy_path:  (T, 33, 3) world 3D coordinates
        output_video_path:      MP4 with skeleton overlay
    """
    # Setup landmarker
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup output video writer
    out_video = None
    if save_video:
        os.makedirs(os.path.dirname(output_video_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    all_landmarks = []
    all_world_landmarks = []
    frames_with_pose = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * 1000.0 / fps)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            frames_with_pose += 1
            pose = result.pose_landmarks[0]

            # Extract normalized landmarks → (33, 5)
            frame_lm = np.array([
                [lm.x, lm.y, lm.z, lm.visibility, lm.presence]
                for lm in pose
            ])

            # Extract world landmarks → (33, 3)
            world_pose = result.pose_world_landmarks[0]
            frame_wlm = np.array([
                [lm.x, lm.y, lm.z]
                for lm in world_pose
            ])

            # Draw skeleton on frame
            if save_video:
                draw_skeleton_on_frame(frame, pose, w, h)
                cv2.putText(
                    frame, f'Frame {frame_idx} | Pose detected',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA
                )
        else:
            frame_lm = np.zeros((33, 5))
            frame_wlm = np.zeros((33, 3))
            if save_video:
                cv2.putText(
                    frame, f'Frame {frame_idx} | No pose',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA
                )

        all_landmarks.append(frame_lm)
        all_world_landmarks.append(frame_wlm)
        if save_video:
            out_video.write(frame)
        frame_idx += 1

    cap.release()
    if out_video is not None:
        out_video.release()
    landmarker.close()

    # Save .npy files
    landmarks_arr = np.array(all_landmarks)        # (T, 33, 5)
    world_arr = np.array(all_world_landmarks)      # (T, 33, 3)

    if save_npy or save_world:
        os.makedirs(os.path.dirname(output_npy_path) or '.', exist_ok=True)
    if save_npy:
        np.save(output_npy_path, landmarks_arr)
    if save_world:
        np.save(output_world_npy_path, world_arr)

    return {
        'landmarks_shape': landmarks_arr.shape,
        'world_shape': world_arr.shape,
        'fps': fps,
        'total_frames': frame_idx,
        'frames_with_pose': frames_with_pose,
    }


# ──────────────────────────────────────────────
# Single video mode
# ──────────────────────────────────────────────
def run_single(video_path: str, model_path: str, output_dir: str,
               save_npy: bool = True, save_world: bool = True, save_video: bool = True):
    stem = Path(video_path).stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = str(out_dir / f"{stem}.npy")
    world_path = str(out_dir / f"{stem}_world.npy")
    video_out = str(out_dir / f"{stem}_skeleton.mp4")

    print(f"Processing: {video_path}")
    print(f"Model:      {model_path}")
    print(f"Output dir: {output_dir}")

    result = process_video(
        video_path, model_path,
        npy_path, world_path, video_out,
        save_npy=save_npy, save_world=save_world, save_video=save_video,
    )

    det_rate = result['frames_with_pose'] / max(result['total_frames'], 1)
    print(f"\nDone!")
    if save_npy:
        print(f"  Landmarks:  {npy_path}  {result['landmarks_shape']}")
    if save_world:
        print(f"  World:      {world_path}  {result['world_shape']}")
    if save_video:
        print(f"  Video:      {video_out}")
    print(f"  Frames:     {result['total_frames']} | Detection: {det_rate:.1%} | FPS: {result['fps']:.1f}")


# ──────────────────────────────────────────────
# Batch mode
# ──────────────────────────────────────────────
def run_batch(input_dir: str, model_path: str, output_dir: str,
              save_npy: bool = True, save_world: bool = True, save_video: bool = True):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_extensions = {'.avi', '.mp4', '.mov', '.mkv'}
    video_files = sorted([
        f for f in input_path.rglob('*')
        if f.suffix.lower() in video_extensions
    ])

    print(f"Found {len(video_files)} videos in {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model:  {model_path}")
    print("=" * 60)

    summary = {'total': len(video_files), 'processed': 0, 'failed': 0, 'results': []}
    start = time.time()

    for i, vf in enumerate(video_files):
        rel = vf.relative_to(input_path)
        out_sub = output_path / rel.parent
        out_sub.mkdir(parents=True, exist_ok=True)

        stem = vf.stem
        npy_path = str(out_sub / f"{stem}.npy")
        world_path = str(out_sub / f"{stem}_world.npy")
        video_out = str(out_sub / f"{stem}_skeleton.mp4")

        # Skip if all outputs exist
        if os.path.exists(npy_path) and os.path.exists(world_path) and os.path.exists(video_out):
            print(f"[{i+1}/{len(video_files)}] SKIP: {rel}")
            summary['processed'] += 1
            continue

        try:
            result = process_video(str(vf), model_path, npy_path, world_path, video_out,
                                   save_npy=save_npy, save_world=save_world, save_video=save_video)
            det = result['frames_with_pose'] / max(result['total_frames'], 1)
            print(
                f"[{i+1}/{len(video_files)}] OK: {rel} "
                f"| {result['total_frames']} frames | {det:.0%} detection"
            )
            summary['processed'] += 1
            summary['results'].append({
                'video': str(rel),
                'frames': result['total_frames'],
                'detection_rate': round(det, 4),
            })
        except Exception as e:
            print(f"[{i+1}/{len(video_files)}] FAIL: {rel} — {e}")
            summary['failed'] += 1
            summary['results'].append({'video': str(rel), 'error': str(e)})

    elapsed = time.time() - start
    summary['elapsed_seconds'] = round(elapsed, 1)

    summary_path = output_path / 'extraction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"Done in {elapsed:.1f}s | Processed: {summary['processed']} | Failed: {summary['failed']}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract skeleton .npy + annotated .mp4 from video using MediaPipe'
    )
    parser.add_argument('--input_dir', type=str, default=None,
                        help='THETIS Video_RGB directory (batch mode)')
    parser.add_argument('--output_dir', type=str, default='./skeleton_output',
                        help='Output directory')
    parser.add_argument('--model', type=str, default='heavy',
                        help='heavy / full / lite, or path to .task file')
    parser.add_argument('--single', type=str, default=None,
                        help='Single video file path')
    parser.add_argument('--no-npy', action='store_true',
                        help='Skip saving normalized landmarks .npy')
    parser.add_argument('--no-world', action='store_true',
                        help='Skip saving world landmarks .npy')
    parser.add_argument('--no-video', action='store_true',
                        help='Skip saving skeleton overlay .mp4')

    args = parser.parse_args()

    # Resolve model
    if args.model.endswith('.task') and os.path.exists(args.model):
        model_path = args.model
    else:
        model_path = get_model_path(args.model)

    save_npy = not args.no_npy
    save_world = not args.no_world
    save_video = not args.no_video

    if args.single:
        run_single(args.single, model_path, args.output_dir,
                   save_npy=save_npy, save_world=save_world, save_video=save_video)
    else:
        if not args.input_dir:
            parser.error("--input_dir is required for batch mode")
        run_batch(args.input_dir, model_path, args.output_dir,
                  save_npy=save_npy, save_world=save_world, save_video=save_video)