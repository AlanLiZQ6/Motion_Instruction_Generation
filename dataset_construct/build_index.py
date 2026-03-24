import os
import csv

TYPE = "canonical"

def parse_filename(filename):
    # remove extension
    name = filename.replace(".npy", "")
    
    # split by "_"
    parts = name.split("_")
    
    subject = parts[0]        # p12
    action_code = parts[1]    # fslice / fflat
    
    # subject id
    subject_id = int(subject[1:])
    level = "beginner" if subject_id <= 31 else "expert"
    
    return subject, action_code, level


def generate_index_csv(beginner_dir, expert_dir, output_name):
    
    rows = []
    
    for file in os.listdir(beginner_dir):
        
        if file.endswith(".npy"):
            path = os.path.join(beginner_dir, file)
            subject, action, level = parse_filename(file)
            if TYPE == "canonical":
                experts_list = os.listdir(expert_dir)
                inference = experts_list[0]
                rows.append([file, path, action, subject, level, inference])

    with open(output_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "path", "action", "subject", "level", "inference"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} samples to {output_name}")