import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
import os



def detect_person_bbox(model, frame):

    # class 0 = person
    results = model(frame, classes=[0], verbose=False)  
    
    # If no person in the frame, return NOne
    if len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best = boxes[np.argmax(areas)]
    return best.astype(int)


def crop_and_resize(frame, bbox, target_size=224, padding=0.2):
    
    
    height, width = frame.shape[:2]
    topleft_x1, topleft_y1, bottom_right_x2, bottom_right_y2 = bbox

    # Add padding around the bounding box
    bw, bh = bottom_right_x2 - topleft_x1, bottom_right_y2 - topleft_y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    topleft_x1 = max(0, topleft_x1 - pad_x)
    topleft_y1 = max(0, topleft_y1 - pad_y)
    bottom_right_x2 = min(width, bottom_right_x2 + pad_x)
    bottom_right_y2 = min(height, bottom_right_y2 + pad_y)

    # Make the crop square (centered on the bbox)
    center_x, center_y = (topleft_x1 + bottom_right_x2) // 2, (topleft_y1 + bottom_right_y2) // 2
    side = max(bottom_right_x2 - topleft_x1, bottom_right_y2 - topleft_y1)
    half = side // 2
    start_x1 = max(0, center_x - half)
    start_y1 = max(0, center_y - half)
    end_x2 = min(width, center_x + half)
    end_y2 = min(height, center_y + half)

    crop = frame[start_y1:end_y2, start_x1:end_x2]
    resized = cv2.resize(crop, (target_size, target_size))
    return resized


def preprocess_video(input_path, output_path, model, target_size=224, padding=0.2):

    obj_cap = cv2.VideoCapture(str(input_path))
    if not obj_cap.isOpened():
        print(f"Error: cannot open {input_path}")
        return False

    fps = obj_cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_size, target_size))

    last_bbox = None
    frame_idx = 0

    while True:
        ret, frame = obj_cap.read()
        if not ret:
            break

        bbox = detect_person_bbox(model, frame)

        if bbox is not None:
            last_bbox = bbox
        elif last_bbox is None:
            # No person detected, just skip this frame
            frame_idx += 1
            continue

        cropped = crop_and_resize(frame, last_bbox, target_size, padding)
        writer.write(cropped)
        frame_idx += 1

    obj_cap.release()
    writer.release()
    print(f"Saved {frame_idx} frames -> {output_path}")
    return True

def main():

    # initiallize the model
    model = YOLO("yolov9c.pt")

    input_path = "/workspace/dataset/VIDEO_RGB"
    output_path = "/workspace/dataset/preprocessed_data"

    
    for dirpath, dirnames, filenames in os.walk(input_path, followlinks=True):
        rel_path = os.path.relpath(dirpath, input_path)
        target_dir = os.path.join(output_path, rel_path)

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
