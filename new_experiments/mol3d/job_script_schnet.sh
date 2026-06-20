#!/bin/bash
#SBATCH -A ki-topml
#SBATCH -p topml
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-24:00:00
#SBATCH --job-name=mol3d_schnet
#SBATCH --output=logs/mol3d_schnet_%j.out
#SBATCH --error=logs/mol3d_schnet_%j.err

source ../../venv/bin/activate

python exp_schnet.py \
    --epochs 300 \
    --batch_size 32 \
    --output results_mol3d_schnet.json
