#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A ki-topml
#SBATCH -p topml
#SBATCH -J rs_ct
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem 64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.12.3-GCCcore-13.3.0
#========[ + + + + Job Steps + + + + ]========#
# Index -> (chem_features, topo_features)
# 0: full  + topo
# 1: full  + no topo
# 2: simple + topo
# 3: simple + no topo
# 4: none  + topo  (topological features only)
CHEM=(full   full   simple simple none)
TOPO=(--topo_features --no-topo_features --topo_features --no-topo_features --topo_features)
NAME=(full_topo full_notopo simple_topo simple_notopo none_topo)

HP_FILE="results/best_hp_ct.json"
HP_ARGS=()
[ -f "$HP_FILE" ] && HP_ARGS=(--hp_file "$HP_FILE")

echo "CPUs: $SLURM_CPUS_PER_TASK | task: ${NAME[$SLURM_ARRAY_TASK_ID]}"
source ../../venv/bin/activate
export PYTHONUNBUFFERED=1
python3 -u exp_ct.py \
    --chem_features "${CHEM[$SLURM_ARRAY_TASK_ID]}" \
    ${TOPO[$SLURM_ARRAY_TASK_ID]} \
    --output "results_ct_${NAME[$SLURM_ARRAY_TASK_ID]}.json" \
    "${HP_ARGS[@]}"
deactivate
