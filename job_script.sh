#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A m2_datamining
#SBATCH -p m2_gpu
#SBATCH -J gpu-solids
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1      # Reserve 1 GPUs
#SBATCH --mem 64G
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.10.4-GCCcore-11.3.0
#========[ + + + + Job Steps + + + + ]========#
models=("TNN" "GCN" "GAN" "GIN")
model=${models[SLURM_ARRAY_TASK_ID]}

source  venv/bin/activate
srun python3 experiment.py --model=$model --path=results --device="cuda"
deactivate