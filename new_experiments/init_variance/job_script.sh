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
#SBATCH --array=0-4
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.12.3-GCCcore-13.3.0
#========[ + + + + Job Steps + + + + ]========#
# 0: TNN
# 1: TNN_Att
# 2: GCN
# 3: GAN
# 4: GIN
MODELS=(TNN TNN_Att GCN GAN GIN)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "CPUs: $SLURM_CPUS_PER_TASK | model: $MODEL"
source ../../venv/bin/activate
export PYTHONUNBUFFERED=1
python3 -u init_variance_exp.py \
    --model $MODEL \
    --num_seeds 100 \
    --device cuda
deactivate
