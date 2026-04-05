import numpy as np
import os
import json


def mediapipe_to_smpl22(mp):
    """
    Convert MediaPipe 33-joint skeleton to SMPL 22-joint format.

    Args:
        mp: (T, 33, 3) MediaPipe landmarks

    Returns:
        smpl: (T, 22, 3) SMPL joints
    """
    T = mp.shape[0]
    smpl = np.zeros((T, 22, 3), dtype=mp.dtype)

    # --- Direct mappings ---
    smpl[:, 1]  = mp[:, 23]   # left_hip
    smpl[:, 2]  = mp[:, 24]   # right_hip
    smpl[:, 4]  = mp[:, 25]   # left_knee
    smpl[:, 5]  = mp[:, 26]   # right_knee
    smpl[:, 7]  = mp[:, 27]   # left_ankle
    smpl[:, 8]  = mp[:, 28]   # right_ankle
    smpl[:, 10] = mp[:, 31]   # left_foot
    smpl[:, 11] = mp[:, 32]   # right_foot
    smpl[:, 15] = mp[:, 0]    # head (nose)
    smpl[:, 16] = mp[:, 11]   # left_shoulder
    smpl[:, 17] = mp[:, 12]   # right_shoulder
    smpl[:, 18] = mp[:, 13]   # left_elbow
    smpl[:, 19] = mp[:, 14]   # right_elbow
    smpl[:, 20] = mp[:, 15]   # left_wrist
    smpl[:, 21] = mp[:, 16]   # right_wrist

    # --- Computed joints ---
    # Pelvis = midpoint of hips
    smpl[:, 0] = (mp[:, 23] + mp[:, 24]) / 2.0

    # Neck = midpoint of shoulders
    neck = (mp[:, 11] + mp[:, 12]) / 2.0
    smpl[:, 12] = neck

    # Spine chain: interpolate between pelvis and neck
    pelvis = smpl[:, 0]
    # spine  (1/3 from pelvis to neck)
    smpl[:, 3] = pelvis + (neck - pelvis) * (1.0 / 3.0)
    # spine1 (2/3 from pelvis to neck)
    smpl[:, 6] = pelvis + (neck - pelvis) * (2.0 / 3.0)
    # spine2 = neck (same as neck, as spine2 connects to neck/collars)
    smpl[:, 9] = neck

    # Left collar  = midpoint of neck and left shoulder
    smpl[:, 13] = (neck + mp[:, 11]) / 2.0
    # Right collar = midpoint of neck and right shoulder
    smpl[:, 14] = (neck + mp[:, 12]) / 2.0

    return smpl


def convert_directory(source_dir, target_dir):
    """Convert all .npy files, mirroring directory structure."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(source_dir):
        rel_path = os.path.relpath(dirpath, source_dir)
        out_dir = os.path.join(target_dir, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        for filename in filenames:
            if not filename.endswith(".npy"):
                continue

            src_path = os.path.join(dirpath, filename)
            dst_path = os.path.join(out_dir, filename)

            mp_data = np.load(src_path)  # (T, 33, 3)
            smpl_data = mediapipe_to_smpl22(mp_data)  # (T, 22, 3)

            np.save(dst_path, smpl_data)
            count += 1
            print(f"[{count}] {src_path} ({mp_data.shape}) -> {dst_path} ({smpl_data.shape})")

    return count


def main():
    params_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "global_params.json")
    )
    with open(params_path) as f:
        params = json.load(f)

    dataset_path = params["dataset_path"]
    source_dir = os.path.join(dataset_path, "skeleton_output")
    target_dir = os.path.join(dataset_path, "skeleton_output_smpl22")

    print(f"Source: {source_dir}  (MediaPipe 33 joints)")
    print(f"Target: {target_dir}  (SMPL 22 joints)\n")

    total = convert_directory(source_dir, target_dir)
    print(f"\nDone. Converted {total} files from 33 -> 22 joints.")


if __name__ == "__main__":
    main()
