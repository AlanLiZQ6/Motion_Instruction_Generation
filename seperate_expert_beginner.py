import os
import shutil

path = "/workspace/dataset/skeleton_output/forehand_flat_skeleton"
expert_dir = os.path.join(path, "experts")
beginner_dir = os.path.join(path, "beginner")

if not os.path.exists(path):
    print(f"Directory not found: {path}")
os.makedirs(expert_dir, exist_ok=True)
os.makedirs(beginner_dir, exist_ok=True)

for file in os.listdir(path):
    file_name, file_type = os.path.splitext(file)
    if (file_type != '.npy'):
        continue
    parts = file_name.split("_")
    data_number = parts[0].replace("p", "")
    src = os.path.join(path, file)
    if int(data_number) > 31:
        shutil.move(src, os.path.join(expert_dir, file))
    else:
        shutil.move(src, os.path.join(beginner_dir, file))

print('All the files has been classified.')