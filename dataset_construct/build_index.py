import os
import csv

TYPE = "canonical"

def parse_filename(filename):
    # remove extension
    name = filename.replace(".npy", "")
    # split by "_"
    parts = name.split("_")
    subject = parts[0]        
    action_code = parts[1]  

    # subject id
    subject_id = int(subject[1:])
    level = "beginner" if subject_id <= 31 else "expert"
    
    return name, subject, action_code, level


def generate_index_csv(npy_beginner_dir, npy_expert_dir, output_name, avi_beginner_dir, avi_expert_dir):
    
    rows = []
    for npy_file in os.listdir(npy_beginner_dir):
        
        if npy_file.endswith(".npy"):
            name, subject, action, level = parse_filename(npy_file)
            beginner_npy_path = os.path.join(npy_beginner_dir, npy_file)
            avi_name = name.replace("_world", "")
            beginner_avi_path = os.path.join(avi_beginner_dir, f'{avi_name}.avi')
            if TYPE == "canonical":
                experts_list = os.listdir(npy_expert_dir)
                inference = experts_list[0]
                inference_name = inference.replace(".npy", "")
                inference_npy_path = os.path.join(npy_expert_dir, inference)
                inference_avi_name = inference_name.replace("_world", "")
                inference_avi_path = os.path.join(avi_expert_dir, f'{inference_avi_name}.avi')
                rows.append([npy_file, action, subject, level, inference, beginner_npy_path, inference_npy_path, beginner_avi_path, inference_avi_path])

    with open(output_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "action", "subject", "level", "inference", "beginner_npy_path", "inference_npy_path", "beginner_avi_path", "inference_avi_path"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} samples to {output_name}")
    
        

