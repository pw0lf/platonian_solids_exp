#!/bin/bash
#SBATCH -A ki-topml
#SBATCH -p topml
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-24:00:00
#SBATCH --job-name=mol3d_ct
#SBATCH --output=logs/mol3d_ct_%A_%a.out
#SBATCH --error=logs/mol3d_ct_%A_%a.err
#SBATCH --array=0-2

source ../../venv/bin/activate

FEAT_MODES=(full simple coords)
FEAT=${FEAT_MODES[$SLURM_ARRAY_TASK_ID]}

python exp_ct.py \
    --feat_mode $FEAT \
    --epochs 300 \
    --batch_size 16 \
    --pe_k 5
