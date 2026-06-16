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
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.12.3-GCCcore-13.3.0
#========[ + + + + Job Steps + + + + ]========#
echo "CPUs: $SLURM_CPUS_PER_TASK"
source  ../venv/bin/activate
export PYTHONUNBUFFERED=1
python3 -u mol3d_exp.py --per_file_size 250000 --epochs 400 --output results_mol3d_ct.json
deactivate
