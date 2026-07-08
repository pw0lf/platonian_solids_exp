#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A ki-topml
#SBATCH -p topml
#SBATCH -J platonic_init_variance
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem 32G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-14
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.12.3-GCCcore-13.3.0
#========[ + + + + Job Steps + + + + ]========#
# 5 models x 3 noise levels (m=120,130,140; eps fixed at 0.3)
# m=150 already done, left out here.
# 0-2:   TNN     @ m=120,130,140
# 3-5:   TNN_Att @ m=120,130,140
# 6-8:   GCN     @ m=120,130,140
# 9-11:  GAN     @ m=120,130,140
# 12-14: GIN     @ m=120,130,140
MODELS=(TNN TNN TNN TNN_Att TNN_Att TNN_Att GCN GCN GCN GAN GAN GAN GIN GIN GIN)
MS=(120 130 140 120 130 140 120 130 140 120 130 140 120 130 140)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
M=${MS[$SLURM_ARRAY_TASK_ID]}

echo "CPUs: $SLURM_CPUS_PER_TASK | model: $MODEL | m: $M"
source ../../venv/bin/activate
export PYTHONUNBUFFERED=1
python3 -u init_variance_exp.py \
    --model $MODEL \
    --num_seeds 100 \
    --device cuda \
    --m $M
deactivate
