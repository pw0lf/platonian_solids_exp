#!/bin/bash
#SBATCH -A ki-topml
#SBATCH -p topml
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-24:00:00
#SBATCH --job-name=mol3d_gnn
#SBATCH --output=logs/mol3d_gnn_%A_%a.out
#SBATCH --error=logs/mol3d_gnn_%A_%a.err
#SBATCH --array=0-2

source ../../venv/bin/activate

MODELS=(GCN GAT GIN)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

python exp_gnn.py \
    --model $MODEL \
    --epochs 300 \
    --batch_size 32 \
    --output results_mol3d_${MODEL,,}.json
