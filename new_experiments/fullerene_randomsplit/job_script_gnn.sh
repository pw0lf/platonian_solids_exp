#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A ki-topml
#SBATCH -p topml
#SBATCH -J rs_gnn
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem 32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.12.3-GCCcore-13.3.0
#========[ + + + + Job Steps + + + + ]========#
# Index -> (model, chem_features)
# 0: GCN + full
# 1: GCN + simple
# 2: GAT + full
# 3: GAT + simple
# 4: GIN + full
# 5: GIN + simple
MODELS=(GCN GCN GAT GAT GIN GIN)
CHEMS=(full simple full simple full simple)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
HP_FILE="results/best_hp_${MODEL,,}.json"
HP_ARGS=()
[ -f "$HP_FILE" ] && HP_ARGS=(--hp_file "$HP_FILE")

echo "CPUs: $SLURM_CPUS_PER_TASK | task: ${MODEL}_${CHEMS[$SLURM_ARRAY_TASK_ID]}"
source ../../venv/bin/activate
export PYTHONUNBUFFERED=1
python3 -u exp_gnn.py \
    --model "$MODEL" \
    --chem_features "${CHEMS[$SLURM_ARRAY_TASK_ID]}" \
    --output "results_${MODEL}_${CHEMS[$SLURM_ARRAY_TASK_ID]}.json" \
    "${HP_ARGS[@]}"
deactivate
