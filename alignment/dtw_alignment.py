"""
DTW alignment on skeleton sequences.

Pipeline:
  1. Load skeleton .npy files (T, 33, 3) for beginner and expert
  2. Compute pairwise Euclidean distance on flattened coordinates
  3. Run standard DTW to find optimal temporal alignment
  4. Write alignment metadata into dataset.json
"""

import numpy as np
import json
import os
from scipy.spatial.distance import cdist


def dtw(query, reference):
    """
    Standard DTW aligning a query (beginner) skeleton sequence
    to a reference (expert) skeleton sequence.

    Args:
        query:     (T1, 33, 3) skeleton array
        reference: (T2, 33, 3) skeleton array

    Returns:
        path: list of (query_idx, ref_idx) pairs
        cost: total DTW cost
        normalized_cost: cost / path_length
    """
    q = query.reshape(len(query), -1)     # (T1, 99)
    r = reference.reshape(len(reference), -1)  # (T2, 99)

    # Pairwise distance matrix
    D = cdist(q, r, metric='euclidean')
    N, M = D.shape

    # Accumulated cost matrix
    C = np.full((N + 1, M + 1), np.inf)
    C[0, 0] = 0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            C[i, j] = D[i - 1, j - 1] + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])

    # Backtrack optimal path
    path = []
    i, j = N, M
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (C[i - 1, j - 1], i - 1, j - 1),
            (C[i - 1, j],     i - 1, j),
            (C[i, j - 1],     i,     j - 1),
        ]
        _, i, j = min(candidates, key=lambda x: x[0])
    path.reverse()

    cost = float(C[N, M])
    return path, cost, cost / len(path)


def align_pair(beginner_path, expert_path):
    """
    Run DTW on a beginner-expert skeleton pair and return alignment metadata.
    """
    beginner = np.load(beginner_path)  # (T1, 33, 3)
    expert = np.load(expert_path)      # (T2, 33, 3)

    path, cost, norm_cost = dtw(beginner, expert)

    beg_frames = [p[0] for p in path]
    exp_frames = [p[1] for p in path]

    return {
        "original_seq_len": beginner.shape[0],
        "aligned_start_frame": min(beg_frames),
        "aligned_end_frame": max(beg_frames),
        "aligned_std_start_frame": min(exp_frames),
        "aligned_std_end_frame": max(exp_frames),
        "aligned_seq_len": len(path),
        "dtw_cost": round(cost, 4),
        "dtw_normalized_cost": round(norm_cost, 4),
    }


def main():
    # Load global params
    params_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "global_params.json")
    )
    with open(params_path) as f:
        params = json.load(f)

    dataset_path = params["dataset_path"]
    skeleton_dir = os.path.join(dataset_path, "skeleton_output_smpl22", "forehand_flat_skeleton")
    beginner_dir = os.path.join(skeleton_dir, "beginner")
    expert_dir = os.path.join(skeleton_dir, "experts")

    # Load dataset.json
    dataset_json_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset.json")
    )
    with open(dataset_json_path) as f:
        dataset = json.load(f)

    # Run DTW alignment for each entry
    for i, entry in enumerate(dataset):
        beg_file = entry["beginner_video_name"]
        exp_file = entry["expert_video_name"]

        beg_path = os.path.join(beginner_dir, beg_file)
        exp_path = os.path.join(expert_dir, exp_file)

        if not os.path.exists(beg_path):
            print(f"[SKIP] Beginner not found: {beg_file}")
            continue
        if not os.path.exists(exp_path):
            print(f"[SKIP] Expert not found: {exp_file}")
            continue

        alignment = align_pair(beg_path, exp_path)
        entry.update(alignment)
        print(
            f"[{i+1}/{len(dataset)}] {beg_file}: "
            f"{alignment['original_seq_len']}fr -> aligned [{alignment['aligned_start_frame']}-{alignment['aligned_end_frame']}], "
            f"expert [{alignment['aligned_std_start_frame']}-{alignment['aligned_std_end_frame']}], "
            f"path_len={alignment['aligned_seq_len']}, norm_cost={alignment['dtw_normalized_cost']:.2f}"
        )

    # Save updated dataset.json
    with open(dataset_json_path, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\nDone. Updated {dataset_json_path} with alignment metadata for {len(dataset)} entries.")


if __name__ == "__main__":
    main()
