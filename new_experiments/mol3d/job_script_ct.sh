#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A ki-topml
#SBATCH -p topml
#SBATCH -J mol3d_ct
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem 64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#========[ + + + + Environment + + + + ]========#
module load lang/Python/3.12.3-GCCcore-13.3.0
#========[ + + + + Job Steps + + + + ]========#
# 0: full (chem + xyz + PE)
# 1: simple (mass/EN/vdW, bond type, ring size)
# 2: coords (3D coords only)
FEAT_MODES=(full simple coords)
FEAT=${FEAT_MODES[$SLURM_ARRAY_TASK_ID]}

echo "CPUs: $SLURM_CPUS_PER_TASK | feat_mode: $FEAT"
source ../../venv/bin/activate
export PYTHONUNBUFFERED=1
python3 -u exp_ct.py \
    --feat_mode $FEAT \
    --epochs 300 \
    --batch_size 16 \
    --pe_k 5
deactivate
