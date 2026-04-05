"""
Visualize Skeleton from .npy files
===================================
Usage:
    # Visualize a single frame (default: frame 0)
    python visualize_skeleton.py --npy skeleton_output/p1_foreflat_s1_world.npy

    # Visualize a specific frame
    python visualize_skeleton.py --npy skeleton_output/p1_foreflat_s1_world.npy --frame 40

    # Save all frames as a video
    python visualize_skeleton.py --npy skeleton_output/p1_foreflat_s1_world.npy --video output.mp4

    # Save all frames as a GIF
    python visualize_skeleton.py --npy skeleton_output/p1_foreflat_s1_world.npy --gif output.gif
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path


# ──────────────────────────────────────────────
# MediaPipe 33 landmark connections (bones)
# ──────────────────────────────────────────────
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),    # left eye → left ear
    (0, 4), (4, 5), (5, 6), (6, 8),    # right eye → right ear
    (9, 10),                             # mouth

    # Torso
    (11, 12),  # shoulders
    (11, 23),  # left shoulder → left hip
    (12, 24),  # right shoulder → right hip
    (23, 24),  # hips

    # Left arm
    (11, 13), (13, 15),  # shoulder → elbow → wrist
    (15, 17), (15, 19), (15, 21),  # wrist → pinky/index/thumb

    # Right arm
    (12, 14), (14, 16),  # shoulder → elbow → wrist
    (16, 18), (16, 20), (16, 22),  # wrist → pinky/index/thumb

    # Left leg
    (23, 25), (25, 27),  # hip → knee → ankle
    (27, 29), (27, 31),  # ankle → heel/foot

    # Right leg
    (24, 26), (26, 28),  # hip → knee → ankle
    (28, 30), (28, 32),  # ankle → heel/foot
]

# Color coding by body part
BONE_COLORS = {
    'torso':     '#2196F3',  # blue
    'left_arm':  '#4CAF50',  # green
    'right_arm': '#FF9800',  # orange
    'left_leg':  '#9C27B0',  # purple
    'right_leg': '#F44336',  # red
    'face':      '#607D8B',  # gray
}

def get_bone_color(i, j):
    """Return color based on which body part a bone belongs to."""
    if i in [11, 12, 23, 24] and j in [11, 12, 23, 24]:
        return BONE_COLORS['torso']
    if i in [11, 13, 15, 17, 19, 21] or j in [11, 13, 15, 17, 19, 21]:
        if not (i in [12, 24, 23] or j in [12, 24, 23]):
            return BONE_COLORS['left_arm']
    if i in [12, 14, 16, 18, 20, 22] or j in [12, 14, 16, 18, 20, 22]:
        if not (i in [11, 23, 24] or j in [11, 23, 24]):
            return BONE_COLORS['right_arm']
    if i in [23, 25, 27, 29, 31] or j in [23, 25, 27, 29, 31]:
        if not (i in [24, 11, 12] or j in [24, 11, 12]):
            return BONE_COLORS['left_leg']
    if i in [24, 26, 28, 30, 32] or j in [24, 26, 28, 30, 32]:
        if not (i in [23, 11, 12] or j in [23, 11, 12]):
            return BONE_COLORS['right_leg']
    return BONE_COLORS['face']


# ──────────────────────────────────────────────
# Key joint labels
# ──────────────────────────────────────────────
KEY_JOINTS = {
    0: 'nose',
    11: 'L.shoulder', 12: 'R.shoulder',
    13: 'L.elbow',    14: 'R.elbow',
    15: 'L.wrist',    16: 'R.wrist',
    23: 'L.hip',      24: 'R.hip',
    25: 'L.knee',     26: 'R.knee',
    27: 'L.ankle',    28: 'R.ankle',
}


# ──────────────────────────────────────────────
# Plot a single frame
# ──────────────────────────────────────────────
def plot_skeleton_frame(
    landmarks: np.ndarray,
    frame_idx: int = 0,
    title: str = None,
    show_labels: bool = True,
    ax: plt.Axes = None,
    elev: float = 15,
    azim: float = -70,
):
    """
    Plot 3D skeleton for one frame.

    Args:
        landmarks: (T, 33, 3) or (33, 3) array of world coordinates
        frame_idx: which frame to plot (if landmarks is 3D)
        title: plot title
        show_labels: whether to label key joints
        ax: existing 3D axes (creates new figure if None)
        elev, azim: camera angle
    """
    if landmarks.ndim == 3:
        frame = landmarks[frame_idx]
    else:
        frame = landmarks

    # Skip if all zeros (no detection)
    if np.allclose(frame, 0):
        print(f"Frame {frame_idx}: no pose detected (all zeros)")
        return None

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        created_fig = True

    ax.clear()

    x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]

    # Plot bones
    for (i, j) in POSE_CONNECTIONS:
        color = get_bone_color(i, j)
        ax.plot(
            [x[i], x[j]], [z[i], z[j]], [-y[i], -y[j]],
            color=color, linewidth=2.5, alpha=0.8,
        )

    # Plot joints
    ax.scatter(x, z, -y, c='black', s=30, zorder=5, depthshade=False)

    # Label key joints
    if show_labels:
        for jid, name in KEY_JOINTS.items():
            ax.text(
                x[jid], z[jid], -y[jid],
                f'  {name}', fontsize=7, color='#333',
                ha='left', va='center',
            )

    # Set axis properties
    center_x = (x.max() + x.min()) / 2
    center_y = (z.max() + z.min()) / 2
    center_z = (-y.max() + -y.min()) / 2
    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2 + 0.1

    ax.set_xlim(center_x - max_range, center_x + max_range)
    ax.set_ylim(center_y - max_range, center_y + max_range)
    ax.set_zlim(center_z - max_range, center_z + max_range)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Y (m)')
    ax.view_init(elev=elev, azim=azim)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Frame {frame_idx}', fontsize=14)

    if created_fig:
        return fig
    return ax


# ──────────────────────────────────────────────
# Save single frame as image
# ──────────────────────────────────────────────
def save_frame_image(landmarks, frame_idx, output_path, show_labels=True):
    """Save a single frame visualization as PNG."""
    fig = plot_skeleton_frame(landmarks, frame_idx, show_labels=show_labels)
    if fig is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved frame {frame_idx} → {output_path}")


# ──────────────────────────────────────────────
# Create video or GIF from all frames
# ──────────────────────────────────────────────
def create_animation(landmarks, output_path, fps=17, show_labels=False):
    """
    Create MP4 video or GIF animation of the full skeleton sequence.

    Args:
        landmarks: (T, 33, 3) world landmarks
        output_path: path ending in .mp4 or .gif
        fps: frames per second
        show_labels: label key joints (can be noisy in video)
    """
    T = landmarks.shape[0]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Compute global axis limits across all frames
    all_valid = landmarks[~np.all(landmarks == 0, axis=(1, 2))]
    if len(all_valid) == 0:
        print("No valid frames found!")
        return

    x_all = all_valid[:, :, 0]
    y_all = all_valid[:, :, 1]
    z_all = all_valid[:, :, 2]

    cx = (x_all.max() + x_all.min()) / 2
    cy = (z_all.max() + z_all.min()) / 2
    cz = (-y_all.max() + -y_all.min()) / 2
    max_range = max(
        x_all.max() - x_all.min(),
        y_all.max() - y_all.min(),
        z_all.max() - z_all.min()
    ) / 2 + 0.15

    def update(frame_idx):
        ax.clear()
        frame = landmarks[frame_idx]

        if np.allclose(frame, 0):
            ax.set_title(f'Frame {frame_idx} (no detection)')
            return

        x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]

        for (i, j) in POSE_CONNECTIONS:
            color = get_bone_color(i, j)
            ax.plot(
                [x[i], x[j]], [z[i], z[j]], [-y[i], -y[j]],
                color=color, linewidth=2.5, alpha=0.8,
            )

        ax.scatter(x, z, -y, c='black', s=25, zorder=5, depthshade=False)

        if show_labels:
            for jid, name in KEY_JOINTS.items():
                ax.text(x[jid], z[jid], -y[jid], f'  {name}', fontsize=6)

        ax.set_xlim(cx - max_range, cx + max_range)
        ax.set_ylim(cy - max_range, cy + max_range)
        ax.set_zlim(cz - max_range, cz + max_range)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_zlabel('Y (m)')
        ax.view_init(elev=15, azim=-70)
        ax.set_title(f'Frame {frame_idx}/{T-1}', fontsize=12)

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps)

    output_path = str(output_path)
    if output_path.endswith('.gif'):
        print(f"Saving GIF ({T} frames)... this may take a minute")
        anim.save(output_path, writer=PillowWriter(fps=fps))
    elif output_path.endswith('.mp4'):
        print(f"Saving MP4 ({T} frames)...")
        anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    else:
        print("Output must end in .gif or .mp4")
        plt.close(fig)
        return

    plt.close(fig)
    print(f"Saved → {output_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize skeleton .npy files')
    parser.add_argument('--npy', type=str, required=True,
                        help='Path to _world.npy file (T, 33, 3)')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to visualize (default: 0)')
    parser.add_argument('--image', type=str, default=None,
                        help='Save single frame as PNG (e.g. frame.png)')
    parser.add_argument('--video', type=str, default=None,
                        help='Save animation as MP4 (e.g. output.mp4)')
    parser.add_argument('--gif', type=str, default=None,
                        help='Save animation as GIF (e.g. output.gif)')
    parser.add_argument('--fps', type=float, default=17,
                        help='FPS for video/gif (default: 17)')
    parser.add_argument('--labels', action='store_true',
                        help='Show joint labels')

    args = parser.parse_args()

    # Load data
    data = np.load(args.npy)
    print(f"Loaded: {args.npy}")
    print(f"Shape: {data.shape}")

    # Handle (T, 33, 5) normalized format — extract x, y, z only
    if data.ndim == 3 and data.shape[2] == 5:
        print("Detected normalized landmarks (T, 33, 5) — using x, y, z columns")
        data = data[:, :, :3]

    if data.ndim == 3 and data.shape[2] == 4:
        print("Detected landmarks (T, 33, 4) — using x, y, z columns")
        data = data[:, :, :3]

    print(f"Frames: {data.shape[0]}")

    if args.video:
        create_animation(data, args.video, fps=args.fps, show_labels=args.labels)
    elif args.gif:
        create_animation(data, args.gif, fps=args.fps, show_labels=args.labels)
    else:
        # Save single frame as PNG
        output = args.image or 'skeleton_frame.png'
        save_frame_image(data, args.frame, output, show_labels=True)