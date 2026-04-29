# Reproduction and Extension: CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation

# Team member: Zhuoqi Li, Raymond Xiang, and Yilin Zheng

# This project is consist of three repositories: Motion_Instruction_Generation, VideoAlignment_forked, and MotionExpert_forked

# For VideoAlignment_forked and MotionExpert_forked, they are revised in the forked version, the coachme model can used for coaching the tennis sports.

# Citations:

# - Yeh et al., "CoachMe: Decoding Sport Elements with a Reformed Coaching Sense,"

# ACL 2025. https://aclanthology.org/2025.acl-long.xxxx/

# - Source repo: https://github.com/MotionXperts/MotionExpert

# Dataset - THETIS

# - Gourgari et al., "Thetis: Three-dimensional tennis shots a human action dataset"

# - Scource repo: https://github.com/THETIS-dataset/dataset

# Reproduce Steps:

# Environment Setup

We use a conda environment named `tennis_coach` with Python 3.10.
PyTorch is installed with the CUDA 12.8 wheels, and the experiment is tested on 5090 GPU devices.

git clone https://github.com/AlanLiZQ6/MotionExpert_forked
git clone https://github.com/AlanLiZQ6/VideoAlignment_forked

# 1. Create and activate the conda environment

conda create -n tennis_coach python=3.10 -y
conda activate tennis_coach
pip install --upgrade pip

# 2. install all the requirements
cd Motion_Instruction_Generation

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

apt-get update && apt-get install -y default-jre

# 3. Download the checkpoints from Hugging Face

Repo: https://huggingface.co/datasets/AlanliQ/Motion_Generation

pip install -U huggingface_hub

### Human3DML pretrain checkpoint
pip install -U gdown

mkdir -p /workspace/MotionExpert_forked/results/pretrain_ref/pretrain_checkpoints

gdown 1oDwh8wWRokey6Payds3YdL3IHcCv5lUF -O /workspace/MotionExpert_forked/results/pretrain_ref/pretrain_checkpoints/checkpoint_epoch_00012.pth

# Download Dataset for Training and Evaluation
mkdir -p /workspace/dataset/pkl_output_smpl22

wget -O /workspace/dataset/pkl_output_smpl22/tennis_standard.pkl https://huggingface.co/datasets/AlanliQ/Motion_Generation/resolve/main/pkl_output_smpl22/tennis_standard.pkl

wget -O /workspace/dataset/pkl_output_smpl22/tennis_test.pkl     https://huggingface.co/datasets/AlanliQ/Motion_Generation/resolve/main/pkl_output_smpl22/tennis_test.pkl

wget -O /workspace/dataset/pkl_output_smpl22/tennis_train.pkl    https://huggingface.co/datasets/AlanliQ/Motion_Generation/resolve/main/pkl_output_smpl22/tennis_train.pkl

# Training Reproduce:
cd /workspace/MotionExpert_forked

### Before Training, you need to check the yaml file and input your pkl dataset path. For example, 
### STANDARD_PATH : "/workspace/dataset/pkl_output_smpl22/tennis_standard.pkl"
### TRAIN : "/workspace/dataset/pkl_output_smpl22/tennis_train.pkl"
### TEST : "/workspace/dataset/pkl_output_smpl22/tennis_test.pkl"

### Train with paper configuration
torchrun --nproc_per_node=1 --master_port=29500 main_tennis.py --cfg_file results/tennis_paper/tennis_paper.yaml


### Train with V2 configuration
torchrun --nproc_per_node=1 --master_port=29500 main_tennis.py --cfg_file results/tennis_v2/tennis_v2.yaml


### Train with V3 configuration
torchrun --nproc_per_node=1 --master_port=29500 main_tennis.py --cfg_file results/tennis_v3/tennis_v3.yaml


### Train with V4 configuration
torchrun --nproc_per_node=1 --master_port=29500 main_tennis.py --cfg_file results/tennis_v4/tennis_v4.yaml


# Evaluation:
cd /workspace/MotionExpert_forked

### paper checkpoint

wget -O results/tennis_paper/checkpoints/checkpoint_epoch_00010.pth https://huggingface.co/datasets/AlanliQ/Motion_Generation/resolve/main/tennis_paper/checkpoint_epoch_00010.pth

### V2 checkpoint

wget -O results/tennis_v2/checkpoints/checkpoint_epoch_00015.pth    https://huggingface.co/datasets/AlanliQ/Motion_Generation/resolve/main/tennis_v2/checkpoint_epoch_00015.pth

### V3 checkpoint

wget -O results/tennis_v3/checkpoints/checkpoint_epoch_00030.pth    https://huggingface.co/datasets/AlanliQ/Motion_Generation/resolve/main/tennis_v3/checkpoint_epoch_00030.pth

### V4 checkpoint
wget -O results/tennis_v4/checkpoints/checkpoint_epoch_00010.pth    https://huggingface.co/datasets/AlanliQ/Motion_Generation/resolve/main/tennis_v4/checkpoint_epoch_00010.pth

### paper configuration evaluation
torchrun --nproc_per_node=1 --master_port=29500 evaluation.py --cfg_file results/tennis_paper/tennis_paper.yaml

### V2 checkpoint
torchrun --nproc_per_node=1 --master_port=29500 evaluation.py --cfg_file results/tennis_v2/tennis_v2.yaml

### V3 checkpoint
torchrun --nproc_per_node=1 --master_port=29500 evaluation.py --cfg_file results/tennis_v3/tennis_v3.yaml

### V4 checkpoint
torchrun --nproc_per_node=1 --master_port=29500 evaluation.py --cfg_file results/tennis_v4/tennis_v4.yaml



