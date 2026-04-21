#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A m2_datamining
#SBATCH -p m2_gpu
#SBATCH -J hp_tuning
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem 64G
#SBATCH --array=0-15
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.10.4-GCCcore-11.3.0
#========[ + + + + Job Steps + + + + ]========#
source ../venv/bin/activate
export PYTHONUNBUFFERED=1

LIFTINGS=(KHop KNN Kernel Cycle)
MODELS=(Simple_Conv Simple_Att Adv_Conv Adv_Att)

LIFTING=${LIFTINGS[$((SLURM_ARRAY_TASK_ID / 4))]}
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID % 4))]}

srun python3 -u hp_tuning_exp.py \
    --path=results_hp_${LIFTING}_${MODEL}.json \
    --lifting=${LIFTING} \
    --model=${MODEL} \
    --datapath=data/data/raw \
    --datasize=100000 \
    --ntrials=20

deactivate
