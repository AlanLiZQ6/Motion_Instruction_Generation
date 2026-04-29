# Reproduction and Extension: CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation

# Team member: Zhuoqi Li, Raymond Xiang, and Yilin Zheng

# This project is consist of three repositories: Motion_Instruction_Generation, VideoAlignment_forked, and MotionExpert_forked

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

# 1. Create and activate the conda environment

conda create -n tennis_coach python=3.10 -y
conda activate tennis_coach
pip install --upgrade pip

# 2. install all the requirements

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# 3.
