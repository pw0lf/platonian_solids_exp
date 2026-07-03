#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A ki-topml
#SBATCH -p topml
#SBATCH -J mf_gnn
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem 32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.12.3-GCCcore-13.3.0
#========[ + + + + Job Steps + + + + ]========#
# 0: GCN  1: GAT  2: GIN
MODELS=(GCN GAT GIN)

echo "CPUs: $SLURM_CPUS_PER_TASK | model: ${MODELS[$SLURM_ARRAY_TASK_ID]}"
source ../../venv/bin/activate
export PYTHONUNBUFFERED=1
python3 -u exp_gnn.py \
    --model "${MODELS[$SLURM_ARRAY_TASK_ID]}" \
    --output "results_${MODELS[$SLURM_ARRAY_TASK_ID]}.json"
deactivate
