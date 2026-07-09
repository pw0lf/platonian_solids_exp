#!/bin/bash
# Submits every hp-tuning job_script_hp_*.sh under new_experiments/ via sbatch.
# fullerene_randomsplit has none (shares fullerene/'s tuning).
# SchNet is excluded -- run locally, not on the cluster.
#
# Usage:
#   ./submit_all_hp_tuning.sh           # submit for real
#   ./submit_all_hp_tuning.sh --dry-run # just list what would be submitted
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

SCRIPTS=(
    "fullerene/job_script_hp_cin.sh"
    "fullerene/job_script_hp_cinpp.sh"
    "fullerene/job_script_hp_ct.sh"
    "fullerene/job_script_hp_fullerenet.sh"
    "fullerene/job_script_hp_gat.sh"
    "fullerene/job_script_hp_gcn.sh"
    "fullerene/job_script_hp_gin.sh"
    "lrgb/peptides_struct/job_script_hp_cin.sh"
    "lrgb/peptides_struct/job_script_hp_cinpp.sh"
    "lrgb/peptides_struct/job_script_hp_ct.sh"
    "lrgb/peptides_struct/job_script_hp_gat.sh"
    "lrgb/peptides_struct/job_script_hp_gcn.sh"
    "lrgb/peptides_struct/job_script_hp_gin.sh"
    "mol3d_fullerene/job_script_hp_cin.sh"
    "mol3d_fullerene/job_script_hp_cinpp.sh"
    "mol3d_fullerene/job_script_hp_ct.sh"
    "mol3d_fullerene/job_script_hp_gat.sh"
    "mol3d_fullerene/job_script_hp_gcn.sh"
    "mol3d_fullerene/job_script_hp_gin.sh"
    "mol3d/job_script_hp_cin.sh"
    "mol3d/job_script_hp_cinpp.sh"
    "mol3d/job_script_hp_ct.sh"
    "mol3d/job_script_hp_gat.sh"
    "mol3d/job_script_hp_gcn.sh"
    "mol3d/job_script_hp_gin.sh"
)

for script in "${SCRIPTS[@]}"; do
    dir=$(dirname "$script")
    name=$(basename "$script")
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] would submit: $script  (cwd=$dir)"
    else
        echo "Submitting: $script  (cwd=$dir)"
        (cd "$dir" && sbatch "$name")
    fi
done

echo "Total hp-tuning job scripts: ${#SCRIPTS[@]}"
