import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
import os



def detect_person_bbox(model, frame):
    """Detect the largest person bounding box in a frame."""
    results = model(frame, classes=[0], verbose=False)  # class 0 = person
    if len(results[0].boxes) == 0:
        return None
    # Pick the largest bounding box by area
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best = boxes[np.argmax(areas)]
    return best.astype(int)  # [x1, y1, x2, y2]


def crop_and_resize(frame, bbox, target_size=224, padding=0.2):
    """Crop around person bbox with padding and resize to target_size x target_size."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Add padding around the bounding box
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    # Make the crop square (centered on the bbox)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    side = max(x2 - x1, y2 - y1)
    half = side // 2
    sx1 = max(0, cx - half)
    sy1 = max(0, cy - half)
    sx2 = min(w, cx + half)
    sy2 = min(h, cy + half)

    crop = frame[sy1:sy2, sx1:sx2]
    resized = cv2.resize(crop, (target_size, target_size))
    return resized


def preprocess_video(input_path, output_path, model, target_size=224, padding=0.2):
    """Detect person per frame, crop around them, resize to 224x224, and save."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: cannot open {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_size, target_size))

    last_bbox = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = detect_person_bbox(model, frame)

        if bbox is not None:
            last_bbox = bbox
        elif last_bbox is None:
            # No person detected yet — skip frame
            frame_idx += 1
            continue

        # Use last known bbox if detection fails this frame
        cropped = crop_and_resize(frame, last_bbox, target_size, padding)
        writer.write(cropped)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved {frame_idx} frames -> {output_path}")
    return True

def main():

    # initiallize the model
    model = YOLO("yolov9c.pt")

    input_path = "/workspace/dataset/VIDEO_RGB"
    output_path = "/workspace/dataset/preprocessed_data"

    # Mirror the VIDEO_RGB directory structure into preprocessed_data,
    # skipping any directories/files that already exist in the output.
    for dirpath, dirnames, filenames in os.walk(input_path, followlinks=True):
        rel_path = os.path.relpath(dirpath, input_path)
        target_dir = os.path.join(output_path, rel_path)

        # Skip subdirectories that already exist in preprocessed_data
        dirnames[:] = [d for d in dirnames
                       if not os.path.isdir(os.path.join(target_dir, d))]

        if not filenames:
            continue

        os.makedirs(target_dir, exist_ok=True)

        for filename in filenames:
            target_file = os.path.join(target_dir, filename)
            if os.path.exists(target_file):
                print(f"Skipping (already exists): {target_file}")
                continue

            source_file = os.path.join(dirpath, filename)
            process_result = preprocess_video(source_file, target_file, model, target_size=224, padding=0.2)
            if process_result == False:
                raise RuntimeError(f"Failed to preprocess video: {source_file}")

        print(f"The videos in the {dirpath} dir have been processed.")

if __name__ == "__main__":
    main()
