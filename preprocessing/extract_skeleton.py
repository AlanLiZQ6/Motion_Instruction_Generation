import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# paths
video_path = "/workspace/dataset/dataset/VIDEO_RGB/forehand_flat/p1_foreflat_s1.avi"
model_path = "/workspace/pose_landmarker_heavy.task"
output_path = "output_skeleton.mp4"

# open video
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w, h)
)

# create model
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
)

landmarker = vision.PoseLandmarker.create_from_options(options)

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
        for lm in result.pose_landmarks[0]:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (0,255,0), -1)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("Saved:", output_path)
