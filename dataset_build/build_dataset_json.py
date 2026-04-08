import json
import os
import pandas as pd
from build_index import generate_index_csv


OUTPUT_PATH = "dataset.json"

def generate_template(beginner_video, expert_video, motion_type):
    """Append a new entry to the JSON dataset file."""
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    entry = {
        "beginner_video_name": beginner_video,
        "expert_video_name": expert_video,
        "motion_type": motion_type,
        "labels": []
    }
    data.append(entry)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Added entry to {os.path.abspath(OUTPUT_PATH)} (total: {len(data)} entries)")


if __name__ == "__main__":
    
    params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "global_params.json"))

    with open(params_path) as f:
        params = json.load(f)
        
    dataset_path = params["dataset_path"]
    skeleton_dir_path = os.path.join(dataset_path, "skeleton_output")
    video_dir_path = os.path.join(dataset_path, "VIDEO_RGB")

    for action_dirs in os.listdir(skeleton_dir_path):
        action_directory = os.path.join(skeleton_dir_path, action_dirs)
        video_action = action_dirs.rsplit("_", 1)[0]
        video_action_directory = os.path.join(video_dir_path, video_action)
        print(action_directory)
        print(video_action_directory)
        sub_directory = action_directory.split("/")
        action_name = sub_directory[-1]
        csv_dir = os.path.join(os.path.dirname(__file__), 'label_csv')
        os.makedirs(csv_dir, exist_ok=True)
        csv_name = os.path.join(csv_dir, f"{action_name}_list.csv")
        for filename in os.listdir(action_directory):
            if filename == 'beginner':
                beginner_dir = os.path.join(action_directory, 'beginner')
                experts_dir = os.path.join(action_directory, 'experts')
                avi_beginner_dir = os.path.join(video_action_directory, 'beginner')
                avi_expert_dir = os.path.join(video_action_directory, 'experts')
                generate_index_csv(beginner_dir,experts_dir,csv_name, avi_beginner_dir, avi_expert_dir)
                index_table = pd.read_csv(csv_name, index_col="id")
                for filename in os.listdir(beginner_dir):
                    if filename in index_table.index:
                        generate_template(filename, index_table.loc[filename, "inference"], index_table.loc[filename, "action"])

