#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A m2_datamining
#SBATCH -p m2_gpu
#SBATCH -J DecEntr
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1      # Reserve 1 GPUs
#SBATCH --mem 64G
#========[ + + + + Environment + + + + ]========#
module load lang/R/4.2.0-foss-2021b
module load lang/Python/3.9.6-GCCcore-11.2.0
module unload lang/SciPy-bundle/2021.10-foss-2021b

#========[ + + + + Job Steps + + + + ]========#
models=("TNN" "GCN" "GAN" "GIN")
model=${models[SLURM_ARRAY_TASK_ID]}

source  ../venv/bin/activate
srun python3 experiment.py --model=$model --path=results --device="cuda"
deactivate