import cv2
import os
import pickle
import random
import json
import torch
from collections import defaultdict


def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def build_samples(source_dir, motion_type):
    
    
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
            name = os.path.join(motion_type, filename.replace(".avi", ""))
            sample = {
                "name": name,
                "video_file": os.path.join(motion_type, sub, filename),
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
    preprocessed_dir = os.path.join(dataset_path, "preprocessed_data")
    output_dir = os.path.join(dataset_path, "tennis_alignment")
    os.makedirs(output_dir, exist_ok=True)

    # Discover all motion types
    motion_types = sorted([
        d for d in os.listdir(preprocessed_dir)
        if os.path.isdir(os.path.join(preprocessed_dir, d))
    ])

    # Create symlinks so VideoAlignment can find videos via
    # PATH_TO_DATASET/<motion_type>/beginner|experts/
    samples = []
    for motion_type in motion_types:
        source_dir = os.path.join(preprocessed_dir, motion_type)
        mt_output = os.path.join(output_dir, motion_type)
        os.makedirs(mt_output, exist_ok=True)
        for sub in ["beginner", "experts"]:
            link = os.path.join(mt_output, sub)
            target = os.path.join(source_dir, sub)
            if not os.path.exists(link):
                os.symlink(target, link)
                print(f"Symlink: {link} -> {target}")

        mt_samples = build_samples(source_dir, motion_type)
        print(f"  {motion_type}: {len(mt_samples)} samples")
        samples.extend(mt_samples)

    print(f"\nTotal samples: {len(samples)}")

    groups = defaultdict(list)
    for s in samples:
        parts = s["video_file"].split(os.sep)
        key = (parts[0], parts[1])
        groups[key].append(s)

    train_set, test_set = [], []
    for key in sorted(groups.keys()):
        group = groups[key]
        random.shuffle(group)
        split = int(len(group) * 0.8)
        train_set.extend(group[:split])
        test_set.extend(group[split:])

    random.shuffle(train_set)
    random.shuffle(test_set)

    # Save
    train_pkl = os.path.join(output_dir, "train.pkl")
    test_pkl = os.path.join(output_dir, "test.pkl")

    with open(train_pkl, "wb") as f:
        pickle.dump(train_set, f)
    with open(test_pkl, "wb") as f:
        pickle.dump(test_set, f)

    print(f"\nTrain dataset in pkl has been finished.")
    print(f"Test dataset in pkl has been finished.")


if __name__ == "__main__":
    main()
