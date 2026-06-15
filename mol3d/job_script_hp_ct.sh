#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A ki-topml
#SBATCH -p topml
#SBATCH -J mol3d
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1      # Reserve 1 GPUs
#SBATCH --mem 64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.12.3-GCCcore-13.3.0
#========[ + + + + Job Steps + + + + ]========#
echo "CPUs: $SLURM_CPUS_PER_TASK"
source  ../venv/bin/activate
export PYTHONUNBUFFERED=1
python3 -u hp_tuning_mol3d_ct.py --output "results_hp_ct_seed${SLURM_ARRAY_TASK_ID}.json" --ntrials 20 --size 5000 --epochs 150 --seed "$((42 + SLURM_ARRAY_TASK_ID))"
deactivate
