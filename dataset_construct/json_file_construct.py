import json
import os
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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
    forehand_flat_skeleton_directory = os.path.join(dataset_path, "skeleton_output", "forehand_flat_skeleton")

    for filename in os.listdir(forehand_flat_skeleton_directory):
        if filename == 'beginner':
            beginner_dir = os.path.join(forehand_flat_skeleton_directory, 'beginner')
            experts_dir = os.path.join(forehand_flat_skeleton_directory, 'experts')
            generate_index_csv(beginner_dir,experts_dir,'test.csv')
            index_table = pd.read_csv("test.csv", index_col="id")
            for filename in os.listdir(beginner_dir):
                if filename in index_table.index:
                    generate_template(filename, index_table.loc[filename, "inference"], index_table.loc[filename, "action"])

