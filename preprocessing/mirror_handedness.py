import os
import cv2


left_Person = {"p5", "p7", "p8", "p19", "p24", "p46", "p48", "p52"}

data_path = os.path.join(os.path.dirname(__file__), "..", "data")
video_dir = os.path.join(data_path, "raw_data")


# Check whether video need to be processed
def check_left(filename: str):
    pid = filename.split("_")[0].lower()
    if pid in left_Person:
        return True
    else:
        return False

# Mirror the frame the write the frame to new video
def mirror_video(path: str):
    
    captured = cv2.VideoCapture(path)
    width  = int(captured.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(captured.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_p_sec    = captured.get(cv2.CAP_PROP_FPS)
    four_char = cv2.VideoWriter_fourcc(*"XVID")

    w_name = path + ".tmp.avi"
    out = cv2.VideoWriter(w_name, four_char, frame_p_sec, (width, height))
    while True:
        ret, frame = captured.read()
        if not ret:
            break
        out.write(cv2.flip(frame, 1))
    captured.release()
    out.release()

    os.replace(w_name, path)


# iterate all the videos
def process_dir(dir: str, type_video: str):
    processed_num = 0
    for dir_path, _, files in os.walk(dir):
        for file_name in sorted(files):
            if not file_name.endswith(type_video):
                continue
            if not check_left(file_name):
                continue
            file_path = os.path.join(dir_path, file_name)
            print(f" Mirroring video: {file_path}")
            mirror_video(file_path)
            print("done")
            processed_num += 1
    return processed_num


def main():
    
    mirror_videos_num = process_dir(video_dir, ".avi")

    print(f"\nProcessed {mirror_videos_num} videos")

if __name__ == "__main__":
    main()
