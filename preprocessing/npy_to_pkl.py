

import numpy as np
import pickle
import os
import json


def convert_npy_to_pkl(source_dir, target_dir):
    """
    Walk through source_dir, convert every .npy file to .pkl,
    and save it under target_dir with the same directory structure.
    """
    count = 0
    for dirpath, dirnames, filenames in os.walk(source_dir):
        # Compute relative path and recreate under target_dir
        rel_path = os.path.relpath(dirpath, source_dir)
        out_dir = os.path.join(target_dir, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        for filename in filenames:
            if not filename.endswith(".npy"):
                continue

            npy_path = os.path.join(dirpath, filename)
            pkl_name = filename.replace(".npy", ".pkl")
            pkl_path = os.path.join(out_dir, pkl_name)

            data = np.load(npy_path)
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)

            count += 1
            print(f"[{count}] {npy_path} -> {pkl_path}  shape={data.shape}")

    return count


def main():
    # Load global params
    params_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "global_params.json")
    )
    with open(params_path) as f:
        params = json.load(f)

    dataset_path = params["dataset_path"]
    source_dir = os.path.join(dataset_path, "skeleton_output_smpl22")
    target_dir = os.path.join(dataset_path, "pkl_output_smpl22")

    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}\n")

    total = convert_npy_to_pkl(source_dir, target_dir)
    print(f"\nDone. Converted {total} files.")


if __name__ == "__main__":
    main()
