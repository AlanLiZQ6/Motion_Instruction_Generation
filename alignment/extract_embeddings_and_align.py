"""
Extract frame-level embeddings from trained VideoAlignment model,
then run DTW alignment using learned embeddings instead of raw skeleton.

Usage (run from /workspace/MotionExpert/VideoAlignment):
    python /workspace/Motion_Instruction_Generation/alignment/extract_embeddings_and_align.py

Or from anywhere with the main conda env active.
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
import pickle
from scipy.spatial.distance import cdist

# Add VideoAlignment to path for model imports
VIDEO_ALIGN_DIR = "/workspace/MotionExpert/VideoAlignment"
sys.path.insert(0, VIDEO_ALIGN_DIR)


def read_video_cv2(video_path):
    """Read video frames as (T, C, H, W) float tensor in [0, 1]."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"Failed to read video: {video_path}")
    video = torch.from_numpy(np.stack(frames))
    video = video.permute(0, 3, 1, 2).float() / 255.0
    return video


def load_model(checkpoint_path, cfg):
    """Load trained CARL model from checkpoint."""
    from model.transformer.transformer import CARL
    model = CARL(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Remove "module." prefix from DDP state dict
    state_dict = {}
    for k, v in ckpt["model_state"].items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        state_dict[new_k] = v
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_embedding(model, video_path, device):
    """Extract per-frame embeddings (T, 128) for a single video."""
    video = read_video_cv2(video_path)  # (T, C, H, W)

    # Normalize like the training pipeline
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    video = (video - mean) / std

    video = video.unsqueeze(0).to(device)  # (1, T, C, H, W)

    with torch.no_grad(), torch.cuda.amp.autocast():
        embs = model(video, video_masks=None, skeleton=None, split="eval")

    return embs[0].cpu().numpy()  # (T, 128)


def dtw_on_embeddings(query_embs, ref_embs):
    """
    Standard DTW on learned embeddings.

    Args:
        query_embs:  (T1, 128) beginner embeddings
        ref_embs:    (T2, 128) expert embeddings

    Returns:
        path, cost, normalized_cost
    """
    D = cdist(query_embs, ref_embs, metric="sqeuclidean")
    N, M = D.shape

    C = np.full((N + 1, M + 1), np.inf)
    C[0, 0] = 0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            C[i, j] = D[i - 1, j - 1] + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])

    # Backtrack
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


def main():
    # --- Config ---
    checkpoint_path = os.path.join(
        VIDEO_ALIGN_DIR,
        "result/tennis_alignment/checkpoints/checkpoint_epoch_00099.pth"
    )
    params_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "global_params.json")
    )
    with open(params_path) as f:
        params = json.load(f)

    dataset_path = params["dataset_path"]
    video_dir = os.path.join(dataset_path, "preprocessed_data", "forehand_flat")
    beginner_dir = os.path.join(video_dir, "beginner")
    expert_dir = os.path.join(video_dir, "experts")

    dataset_json_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset.json")
    )

    # --- Load model ---
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, cfg)
    model = model.to(device)
    print(f"Loaded model from {checkpoint_path} (epoch {ckpt['epoch']})")
    print(f"Device: {device}")

    # --- Cache embeddings (avoid recomputing for shared experts) ---
    embedding_cache = {}

    def get_embedding(video_path):
        if video_path not in embedding_cache:
            embedding_cache[video_path] = extract_embedding(model, video_path, device)
        return embedding_cache[video_path]

    # --- Load dataset.json ---
    with open(dataset_json_path) as f:
        dataset = json.load(f)

    # --- Run alignment ---
    for i, entry in enumerate(dataset):
        beg_name = entry["beginner_video_name"].replace("_world.npy", ".avi")
        exp_name = entry["expert_video_name"].replace("_world.npy", ".avi")

        beg_path = os.path.join(beginner_dir, beg_name)
        exp_path = os.path.join(expert_dir, exp_name)

        if not os.path.exists(beg_path):
            print(f"[SKIP] Beginner not found: {beg_name}")
            continue
        if not os.path.exists(exp_path):
            print(f"[SKIP] Expert not found: {exp_name}")
            continue

        beg_embs = get_embedding(beg_path)
        exp_embs = get_embedding(exp_path)

        path, cost, norm_cost = dtw_on_embeddings(beg_embs, exp_embs)

        beg_frames = [p[0] for p in path]
        exp_frames = [p[1] for p in path]

        entry["aligned_start_frame"] = min(beg_frames)
        entry["aligned_end_frame"] = max(beg_frames)
        entry["aligned_std_start_frame"] = min(exp_frames)
        entry["aligned_std_end_frame"] = max(exp_frames)
        entry["aligned_seq_len"] = len(path)
        entry["dtw_cost"] = round(cost, 4)
        entry["dtw_normalized_cost"] = round(norm_cost, 4)
        entry["original_seq_len"] = beg_embs.shape[0]

        print(
            f"[{i+1}/{len(dataset)}] {beg_name}: "
            f"{beg_embs.shape[0]}fr -> aligned [{entry['aligned_start_frame']}-{entry['aligned_end_frame']}], "
            f"expert [{entry['aligned_std_start_frame']}-{entry['aligned_std_end_frame']}], "
            f"path_len={entry['aligned_seq_len']}, norm_cost={norm_cost:.4f}"
        )

    # --- Save ---
    with open(dataset_json_path, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\nDone. Updated {dataset_json_path} with embedding-based alignment for {len(dataset)} entries.")


if __name__ == "__main__":
    main()
