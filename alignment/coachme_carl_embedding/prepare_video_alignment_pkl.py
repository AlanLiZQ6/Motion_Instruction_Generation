"""
Prepare train.pkl and test.pkl for VideoAlignment training.

Reads all .avi files from the preprocessed tennis dataset,
extracts frame counts, and splits into train/test sets (80/20).

Output directory structure:
    /workspace/dataset/tennis_alignment/
    ├── train.pkl
    ├── test.pkl
    ├── beginner/ -> symlink to source
    └── experts/  -> symlink to source

Each sample in the pkl:
    {
        "name":        str,           # e.g. "p25_foreflat_s2"
        "video_file":  str,           # e.g. "beginner/p25_foreflat_s2.avi"
        "frame_label": Tensor(seq_len),  # zeros (placeholder)
        "seq_len":     int,           # total frames
    }
"""

import cv2
import os
import pickle
import random
import json
import torch


def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def build_samples(source_dir):
    """Scan beginner/ and experts/ under source_dir, return list of sample dicts."""
    samples = []
    for sub in ["beginner", "experts"]:
        sub_dir = os.path.join(source_dir, sub)
        if not os.path.isdir(sub_dir):
            print(f"[WARN] Directory not found: {sub_dir}")
            continue
        for filename in sorted(os.listdir(sub_dir)):
            if not filename.endswith(".avi"):
                continue
            video_path = os.path.join(sub_dir, filename)
            seq_len = get_frame_count(video_path)
            if seq_len == 0:
                print(f"[SKIP] 0 frames: {video_path}")
                continue
            name = filename.replace(".avi", "")
            sample = {
                "name": name,
                "video_file": os.path.join(sub, filename),
                "frame_label": torch.zeros(seq_len),
                "seq_len": seq_len,
            }
            samples.append(sample)
    return samples


def main():
    params_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "global_params.json")
    )
    with open(params_path) as f:
        params = json.load(f)

    dataset_path = params["dataset_path"]
    source_dir = os.path.join(dataset_path, "preprocessed_data", "forehand_flat")
    output_dir = os.path.join(dataset_path, "tennis_alignment")
    os.makedirs(output_dir, exist_ok=True)

    # Create symlinks so VideoAlignment can find videos via PATH_TO_DATASET
    for sub in ["beginner", "experts"]:
        link = os.path.join(output_dir, sub)
        target = os.path.join(source_dir, sub)
        if not os.path.exists(link):
            os.symlink(target, link)
            print(f"Symlink: {link} -> {target}")

    # Build all samples
    samples = build_samples(source_dir)
    print(f"\nTotal samples: {len(samples)}")

    # Split train/test (80/20), stratified by beginner/expert
    random.seed(42)
    beginners = [s for s in samples if "beginner" in s["video_file"]]
    experts = [s for s in samples if "experts" in s["video_file"]]
    random.shuffle(beginners)
    random.shuffle(experts)

    split_b = int(len(beginners) * 0.8)
    split_e = int(len(experts) * 0.8)

    train_set = beginners[:split_b] + experts[:split_e]
    test_set = beginners[split_b:] + experts[split_e:]
    random.shuffle(train_set)
    random.shuffle(test_set)

    # Save
    train_pkl = os.path.join(output_dir, "train.pkl")
    test_pkl = os.path.join(output_dir, "test.pkl")

    with open(train_pkl, "wb") as f:
        pickle.dump(train_set, f)
    with open(test_pkl, "wb") as f:
        pickle.dump(test_set, f)

    print(f"\nTrain: {len(train_set)} samples -> {train_pkl}")
    print(f"Test:  {len(test_set)} samples -> {test_pkl}")

    # Print a few examples
    print("\n--- Train sample example ---")
    s = train_set[0]
    print(f"  name: {s['name']}")
    print(f"  video_file: {s['video_file']}")
    print(f"  seq_len: {s['seq_len']}")
    print(f"  frame_label shape: {s['frame_label'].shape}")

    print("\n--- Test sample example ---")
    s = test_set[0]
    print(f"  name: {s['name']}")
    print(f"  video_file: {s['video_file']}")
    print(f"  seq_len: {s['seq_len']}")
    print(f"  frame_label shape: {s['frame_label'].shape}")


if __name__ == "__main__":
    main()
