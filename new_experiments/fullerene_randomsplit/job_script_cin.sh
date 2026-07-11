#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A ki-topml
#SBATCH -p topml
#SBATCH -J rs_cin
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem 64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.12.3-GCCcore-13.3.0
#========[ + + + + Job Steps + + + + ]========#
# 0: CIN   (up + boundary messages)
# 1: CINpp (up + down + boundary messages)
MODELS=(CIN CINpp)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
HP_FILE="results/best_hp_${MODEL,,}.json"
HP_ARGS=()
[ -f "$HP_FILE" ] && HP_ARGS=(--hp_file "$HP_FILE")

echo "CPUs: $SLURM_CPUS_PER_TASK | model: $MODEL"
source ../../venv/bin/activate
export PYTHONUNBUFFERED=1
python3 -u exp_cin.py \
    --model $MODEL \
    --epochs 300 \
    --batch_size 32 \
    --hidden 128 \
    --num_layers 4 \
    "${HP_ARGS[@]}"
deactivate
