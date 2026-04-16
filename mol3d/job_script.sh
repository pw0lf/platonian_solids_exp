#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A m2_datamining
#SBATCH -p m2_gpu
#SBATCH -J schnet
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1      # Reserve 1 GPUs
#SBATCH --mem 64G
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.10.4-GCCcore-11.3.0
#========[ + + + + Job Steps + + + + ]========#
source  ../venv/bin/activate
export PYTHONUNBUFFERED=1
srun python3 -u tnn.py --path=result_tnn3.json
deactivate