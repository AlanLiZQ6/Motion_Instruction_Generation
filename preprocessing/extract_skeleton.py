import cv2
import mediapipe as mp
import numpy as np
import json
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision



def extract_skeleton(video_path, model_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    all_frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((frame_idx / fps) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            frame_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        else:
            frame_data = np.zeros((33, 3))

        all_frames.append(frame_data)
        frame_idx += 1

    cap.release()

    skeleton = np.array(all_frames) 
    np.save(output_path, skeleton)
    print(f"Saved: {output_path}  shape: {skeleton.shape}")

def main():
    params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "global_params.json"))

    with open(params_path) as f:
        params = json.load(f)

    dataset_path = params["dataset_path"]
    output_base = os.path.join(dataset_path, "skeleton_output")
    input_path = os.path.join(dataset_path, "VIDEO_RGB")
    model_path = params["MediaPipe_model_path"]

    os.makedirs(output_base, exist_ok=True)

    for action_dir in sorted(os.listdir(input_path)):
        action_input = os.path.join(input_path, action_dir)
        if not os.path.isdir(action_input):
            continue

        skeleton_dir_name = action_dir + "_skeleton"
        action_output = os.path.join(output_base, skeleton_dir_name)

        if os.path.isdir(action_output):
            print(f"Skipping {action_dir}: {skeleton_dir_name} already exists")
            continue

        for root, _, files in os.walk(action_input):
            for filename in sorted(files):
                if not filename.endswith(".avi"):
                    continue

                video_file = os.path.join(root, filename)
                rel_path = os.path.relpath(root, action_input)
                npy_dir = os.path.join(action_output, rel_path)
                os.makedirs(npy_dir, exist_ok=True)

                npy_name = os.path.splitext(filename)[0] + ".npy"
                npy_file = os.path.join(npy_dir, npy_name)

                print(f"Processing: {video_file}")
                extract_skeleton(video_file, model_path, npy_file)

if __name__ == "__main__":
    
    main()
