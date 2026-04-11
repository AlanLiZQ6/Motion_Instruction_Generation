"""
Build CoachMe-format pickle files from dataset.json + skeleton .npy files.

Produces:
  - pkl_output_smpl22/tennis_train.pkl   (training samples, ~80%)
  - pkl_output_smpl22/tennis_test.pkl    (test samples, ~20%)
  - pkl_output_smpl22/tennis_standard.pkl (one expert reference per motion type)

Each pkl file contains a list of dicts matching CoachMe's expected format.
Supports all 12 motion types. Train/test split is stratified by
(motion_type, person) and shuffled to ensure balanced batches.
"""

import numpy as np
import pickle
import json
import os
import random
from collections import defaultdict


# Map short motion_type names in dataset.json to skeleton directory names
MOTION_TYPE_TO_DIR = {
    "backhand": "backhand_skeleton",
    "backhand2h": "backhand2hands_skeleton",
    "bslice": "backhand_slice_skeleton",
    "bvolley": "backhand_volley_skeleton",
    "foreflat": "forehand_flat_skeleton",
    "foreopen": "forehand_openstands_skeleton",
    "fslice": "forehand_slice_skeleton",
    "fvolley": "forehand_volley_skeleton",
    "serflat": "flat_service_skeleton",
    "serkick": "kick_service_skeleton",
    "serslice": "slice_service_skeleton",
    "smash": "smash_skeleton",
}


def main():
    random.seed(42)

    # Load global params
    params_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "global_params.json")
    )
    with open(params_path) as f:
        params = json.load(f)

    dataset_path = params["dataset_path"]
    smpl22_dir = os.path.join(dataset_path, "skeleton_output_smpl22")
    output_dir = os.path.join(dataset_path, "pkl_output_smpl22")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset.json
    dataset_json_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "dataset.json")
    )
    with open(dataset_json_path) as f:
        dataset = json.load(f)

    # ---------------------------------------------------------------
    # 1. Build standard (expert) pickle — one expert per motion type
    # ---------------------------------------------------------------
    # Collect unique expert per motion type
    expert_map = {}
    for entry in dataset:
        mt = entry["motion_type"]
        if mt not in expert_map:
            expert_map[mt] = entry["expert_video_name"]

    standard_list = []
    expert_skeletons = {}  # mt -> skeleton array, for alignment clamping
    for mt, expert_name in sorted(expert_map.items()):
        skel_dir = os.path.join(smpl22_dir, MOTION_TYPE_TO_DIR[mt], "experts")
        expert_path = os.path.join(skel_dir, expert_name)
        if not os.path.exists(expert_path):
            print(f"[WARN] Expert not found: {expert_path}")
            continue
        expert_skel = np.load(expert_path)  # (T, 22, 3)
        expert_skeletons[mt] = expert_skel
        standard_list.append({
            "video_name": mt,
            "motion_type": mt,
            "coordinates": expert_skel,
        })
        print(f"Expert [{mt}]: {expert_name}  shape={expert_skel.shape}")

    standard_path = os.path.join(output_dir, "tennis_standard.pkl")
    with open(standard_path, "wb") as f:
        pickle.dump(standard_list, f)
    print(f"\nSaved standard: {standard_path}  ({len(standard_list)} experts)")

    # ---------------------------------------------------------------
    # 2. Build beginner sample list — all motion types
    # ---------------------------------------------------------------
    samples = []
    skip_count = 0
    for entry in dataset:
        mt = entry["motion_type"]
        dir_name = MOTION_TYPE_TO_DIR[mt]
        beg_name = entry["beginner_video_name"]
        beg_path = os.path.join(smpl22_dir, dir_name, "beginner", beg_name)

        if not os.path.exists(beg_path):
            print(f"[SKIP] Not found: {beg_path}")
            skip_count += 1
            continue

        beg_skel = np.load(beg_path)  # (T, 22, 3)
        video_name = beg_name.replace(".npy", "")

        beg_len = beg_skel.shape[0]
        std_len = expert_skeletons[mt].shape[0] if mt in expert_skeletons else beg_len

        # Use alignment results from dataset.json (computed by DTW)
        aligned_start = entry.get("aligned_start_frame", 0)
        aligned_end = entry.get("aligned_end_frame", beg_len - 1)
        aligned_std_start = entry.get("aligned_std_start_frame", 0)
        aligned_std_end = entry.get("aligned_std_end_frame", std_len - 1)
        aligned_seq_len = entry.get("aligned_seq_len", min(beg_len, std_len))

        # Clamp to valid ranges
        aligned_seq_len = min(aligned_seq_len, beg_len - aligned_start, std_len - aligned_std_start)

        sample = {
            "video_name": video_name,
            "motion_type": mt,
            "coordinates": beg_skel,
            "labels": entry["labels"],
            "augmented_labels": entry.get("augmented_labels", None),
            "original_seq_len": beg_len,
            "aligned_start_frame": aligned_start,
            "aligned_end_frame": aligned_end,
            "aligned_std_start_frame": aligned_std_start,
            "aligned_std_end_frame": aligned_std_end,
            "aligned_seq_len": aligned_seq_len,
        }
        samples.append(sample)

    print(f"\nTotal samples: {len(samples)}  (skipped: {skip_count})")

    # ---------------------------------------------------------------
    # 3. Train/test split (80/20, stratified by motion_type + person)
    # ---------------------------------------------------------------
    # Group by (motion_type, person_id) to ensure balanced split
    groups = defaultdict(list)
    for s in samples:
        person_id = s["video_name"].split("_")[0]
        groups[(s["motion_type"], person_id)].append(s)

    train_samples, test_samples = [], []
    for key in sorted(groups.keys()):
        group = groups[key]
        random.shuffle(group)
        split = int(len(group) * 0.8)
        if split == 0 and len(group) > 0:
            split = 1  # ensure at least 1 sample in train
        train_samples.extend(group[:split])
        test_samples.extend(group[split:])

    # Shuffle to mix motion types across batches
    random.shuffle(train_samples)
    random.shuffle(test_samples)

    # Print distribution
    train_dist = defaultdict(int)
    test_dist = defaultdict(int)
    for s in train_samples:
        train_dist[s["motion_type"]] += 1
    for s in test_samples:
        test_dist[s["motion_type"]] += 1

    print(f"\nTrain: {len(train_samples)} samples")
    print(f"Test:  {len(test_samples)} samples")
    print(f"\nPer motion type:")
    print(f"  {'motion_type':<15} {'train':>6} {'test':>6}")
    for mt in sorted(set(list(train_dist.keys()) + list(test_dist.keys()))):
        print(f"  {mt:<15} {train_dist[mt]:>6} {test_dist[mt]:>6}")

    # ---------------------------------------------------------------
    # 4. Save train/test pickles
    # ---------------------------------------------------------------
    train_path = os.path.join(output_dir, "tennis_train.pkl")
    with open(train_path, "wb") as f:
        pickle.dump(train_samples, f)
    print(f"\nSaved: {train_path}")

    test_path = os.path.join(output_dir, "tennis_test.pkl")
    with open(test_path, "wb") as f:
        pickle.dump(test_samples, f)
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
