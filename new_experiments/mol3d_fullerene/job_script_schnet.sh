#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A ki-topml
#SBATCH -p topml
#SBATCH -J mf_schnet
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem 32G
#SBATCH --cpus-per-task=4
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.12.3-GCCcore-13.3.0
#========[ + + + + Job Steps + + + + ]========#
HP_FILE="results/best_hp_schnet.json"
HP_ARGS=()
[ -f "$HP_FILE" ] && HP_ARGS=(--hp_file "$HP_FILE")

echo "CPUs: $SLURM_CPUS_PER_TASK"
source ../../venv/bin/activate
export PYTHONUNBUFFERED=1
python3 -u exp_schnet.py --output "results_schnet.json" "${HP_ARGS[@]}"
deactivate
