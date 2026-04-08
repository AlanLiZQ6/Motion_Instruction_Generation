import os
import shutil
import json


def separate_expert_beginner(path):
    
    
    expert_dir = os.path.join(path, "experts")
    beginner_dir = os.path.join(path, "beginner")

    if not os.path.exists(path):
        print(f"Directory not found: {path}")
    os.makedirs(expert_dir, exist_ok=True)
    os.makedirs(beginner_dir, exist_ok=True)

    for file in os.listdir(path):
        file_name, file_type = os.path.splitext(file)
        if (file_type != '.npy') and (file_type != '.avi'):
            continue
        parts = file_name.split("_")
        data_number = parts[0].replace("p", "")
        src = os.path.join(path, file)
        if int(data_number) > 31:
            shutil.move(src, os.path.join(expert_dir, file))
        else:
            shutil.move(src, os.path.join(beginner_dir, file))

    print(f'{path} has been classified.')
    


def main():
    
    # Access the global parameters
    params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "global_params.json"))

    with open(params_path) as f:
        params = json.load(f)
        
    # Access the dataset directory 
    dataset_path = params["dataset_path"]

    for data_dirs in os.listdir(dataset_path):
        data_type = os.path.join(dataset_path, data_dirs)
        if not os.path.isdir(data_type):
            continue
        for action_dirs in os.listdir(data_type):
            action_path = os.path.join(data_type, action_dirs)
            if not os.path.isdir(action_path):
                continue
            separate_expert_beginner(action_path)


    print('All the files has been classified !')


if __name__ == "__main__":
    main()
