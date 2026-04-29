import numpy as np
import os
import json


def mediapipe_to_smpl_v22(media_pipe_v):

    T = media_pipe_v.shape[0]
    smpl_v = np.zeros((T, 22, 3), dtype=media_pipe_v.dtype)

    smpl_v[:, 1]  = media_pipe_v[:, 23]   # left_hip
    smpl_v[:, 2]  = media_pipe_v[:, 24]   # right_hip
    smpl_v[:, 4]  = media_pipe_v[:, 25]   # left_knee
    smpl_v[:, 5]  = media_pipe_v[:, 26]   # right_knee
    smpl_v[:, 7]  = media_pipe_v[:, 27]   # left_ankle
    smpl_v[:, 8]  = media_pipe_v[:, 28]   # right_ankle
    smpl_v[:, 10] = media_pipe_v[:, 31]   # left_foot
    smpl_v[:, 11] = media_pipe_v[:, 32]   # right_foot
    smpl_v[:, 15] = media_pipe_v[:, 0]    # head (nose)
    smpl_v[:, 16] = media_pipe_v[:, 11]   # left_shoulder
    smpl_v[:, 17] = media_pipe_v[:, 12]   # right_shoulder
    smpl_v[:, 18] = media_pipe_v[:, 13]   # left_elbow
    smpl_v[:, 19] = media_pipe_v[:, 14]   # right_elbow
    smpl_v[:, 20] = media_pipe_v[:, 15]   # left_wrist
    smpl_v[:, 21] = media_pipe_v[:, 16]   # right_wrist

    # Pelvis is the midpoint of hips
    smpl_v[:, 0] = (media_pipe_v[:, 23] + media_pipe_v[:, 24]) / 2.0
    # Neck is the midpoint of shoulders
    smpl_v[:, 12] = (media_pipe_v[:, 11] + media_pipe_v[:, 12]) / 2.0
    # Left collar is the midpoint of neck and left shoulder
    smpl_v[:, 13] = (smpl_v[:, 12] + media_pipe_v[:, 11]) / 2.0
    # Right collar is the midpoint of neck and right shoulder
    smpl_v[:, 14] = (smpl_v[:, 12] + media_pipe_v[:, 12]) / 2.0
    # spine should be equal to neck
    smpl_v[:, 9] = (media_pipe_v[:, 11] + media_pipe_v[:, 12]) / 2.0
    # spine1 can be set to 1/3 from pelvis to neck)
    smpl_v[:, 3] = smpl_v[:, 0] + (smpl_v[:, 12] - smpl_v[:, 0]) * (1.0 / 3.0)
    # spine2 can be set to 2/3 from pelvis ot neck
    smpl_v[:, 6] = smpl_v[:, 0] + (smpl_v[:, 12] - smpl_v[:, 0]) * (2.0 / 3.0)

    return smpl_v


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

            mp_data = np.load(src_path)  
            smpl_v_data = mediapipe_to_smpl_v22(mp_data)

            np.save(dst_path, smpl_v_data)
            count += 1
            print(f"[{count}] {src_path} ({mp_data.shape}) -> {dst_path} ({smpl_v_data.shape})")

    return count


def main():
    params_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "global_params.json")
    )
    with open(params_path) as f:
        params = json.load(f)

    dataset_path = params["dataset_path"]
    source_dir = os.path.join(dataset_path, "skeleton_output")
    target_dir = os.path.join(dataset_path, "skeleton_output_smpl_v22")

    print(f"Source: {source_dir}  (MediaPipe 33 joints)")
    print(f"Target: {target_dir}  (smpl_v 22 joints)\n")

    total = convert_directory(source_dir, target_dir)
    print(f"\nDone. Converted {total} files from 33 -> 22 joints.")


if __name__ == "__main__":
    main()
