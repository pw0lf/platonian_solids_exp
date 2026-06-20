#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A ki-topml
#SBATCH -p topml
#SBATCH -J struct_gnn
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem 32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.12.3-GCCcore-13.3.0
#========[ + + + + Job Steps + + + + ]========#
# 0: GCN
# 1: GAT
# 2: GIN
MODELS=(GCN GAT GIN)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "CPUs: $SLURM_CPUS_PER_TASK | model: $MODEL"
source ../../venv/bin/activate
export PYTHONUNBUFFERED=1
python3 -u exp_gnn.py \
    --model $MODEL \
    --epochs 300 \
    --batch_size 32
deactivate
