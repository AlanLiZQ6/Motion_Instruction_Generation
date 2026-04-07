"""
Build CoachMe-format pickle files from dataset.json + skeleton .npy files.

Produces:
  - pkl_output/tennis_train.pkl   (training samples, ~80%)
  - pkl_output/tennis_test.pkl    (test samples, ~20%)
  - pkl_output/tennis_standard.pkl (expert reference skeleton)

Each pkl file contains a list of dicts matching CoachMe's expected format.
"""

import numpy as np
import pickle
import json
import os
import random


def main():
    random.seed(42)

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
    output_dir = os.path.join(dataset_path, "pkl_output_smpl22")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset.json
    dataset_json_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "dataset.json")
    )
    with open(dataset_json_path) as f:
        dataset = json.load(f)

    # ---------------------------------------------------------------
    # 1. Build standard (expert) pickle
    # ---------------------------------------------------------------
    # All entries use the same expert: p46_foreflat_s1_world.npy
    expert_name = dataset[0]["expert_video_name"]
    expert_skel = np.load(os.path.join(expert_dir, expert_name))  # (T, 33, 3)

    standard_list = [{
        "video_name": "foreflat",
        "motion_type": "foreflat",
        "coordinates": expert_skel,
    }]

    standard_path = os.path.join(output_dir, "tennis_standard.pkl")
    with open(standard_path, "wb") as f:
        pickle.dump(standard_list, f)
    print(f"Saved standard: {standard_path}  (expert: {expert_name}, shape: {expert_skel.shape})")

    # ---------------------------------------------------------------
    # 2. Build beginner sample list
    # ---------------------------------------------------------------
    samples = []
    for entry in dataset:
        beg_name = entry["beginner_video_name"]
        beg_path = os.path.join(beginner_dir, beg_name)

        if not os.path.exists(beg_path):
            print(f"[SKIP] Not found: {beg_name}")
            continue

        beg_skel = np.load(beg_path)  # (T, 33, 3)

        # Video name without extension (matches CoachMe convention)
        video_name = beg_name.replace("_world.npy", "")

        beg_len = beg_skel.shape[0]
        std_len = expert_skel.shape[0]

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
            "motion_type": entry["motion_type"],
            "coordinates": beg_skel,
            "labels": entry["labels"],
            "augmented_labels": None,
            "original_seq_len": beg_len,
            "aligned_start_frame": aligned_start,
            "aligned_end_frame": aligned_end,
            "aligned_std_start_frame": aligned_std_start,
            "aligned_std_end_frame": aligned_std_end,
            "aligned_seq_len": aligned_seq_len,
        }
        samples.append(sample)

    print(f"\nTotal samples: {len(samples)}")

    # ---------------------------------------------------------------
    # 3. Train/test split (80/20, grouped by person)
    # ---------------------------------------------------------------
    # Group by person ID to avoid data leakage
    person_samples = {}
    for s in samples:
        # Extract person ID: p10_foreflat_s1 -> p10
        person_id = s["video_name"].split("_")[0]
        if person_id not in person_samples:
            person_samples[person_id] = []
        person_samples[person_id].append(s)

    person_ids = sorted(person_samples.keys())
    random.shuffle(person_ids)

    split_idx = int(len(person_ids) * 0.8)
    train_persons = set(person_ids[:split_idx])
    test_persons = set(person_ids[split_idx:])

    train_samples = [s for pid in train_persons for s in person_samples[pid]]
    test_samples = [s for pid in test_persons for s in person_samples[pid]]

    print(f"Train: {len(train_samples)} samples ({len(train_persons)} persons: {sorted(train_persons)})")
    print(f"Test:  {len(test_samples)} samples ({len(test_persons)} persons: {sorted(test_persons)})")

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
