import os
import csv

DATA_DIR = "data"
OUTPUT = "index.csv"

def parse_filename(filename):
    # remove extension
    name = filename.replace(".avi", "")
    
    # split by "_"
    parts = name.split("_")
    
    subject = parts[0]        # p12
    action_code = parts[1]    # fslice / fflat
    
    # map action
    if action_code == "fslice":
        action = "forehand_slice"
    elif action_code == "fflat":
        action = "forehand_flat"
    else:
        action = action_code  # fallback
    
    # subject id
    subject_id = int(subject[1:])
    level = "beginner" if subject_id <= 31 else "expert"
    
    return subject, action, level

rows = []

for action_folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, action_folder)
    
    if not os.path.isdir(folder_path):
        continue
    
    for file in os.listdir(folder_path):
        if file.endswith(".avi"):
            path = os.path.join(folder_path, file)
            
            subject, action, level = parse_filename(file)
            
            rows.append([file, path, action, subject, level])

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "path", "action", "subject", "level"])
    writer.writerows(rows)

print(f"Saved {len(rows)} samples to {OUTPUT}")