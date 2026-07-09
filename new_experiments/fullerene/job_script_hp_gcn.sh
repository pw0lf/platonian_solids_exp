#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A ki-topml
#SBATCH -p topml
#SBATCH -J fullerene_hp_gcn
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem 32G
#SBATCH --cpus-per-task=4
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.12.3-GCCcore-13.3.0
#========[ + + + + Job Steps + + + + ]========#
echo "CPUs: $SLURM_CPUS_PER_TASK | hp tuning: GCN"
source ../../venv/bin/activate
export PYTHONUNBUFFERED=1
python3 -u hp_tuning_gcn.py \
    --n_trials 10 \
    --epochs 30 \
    --batch_size 32
deactivate
