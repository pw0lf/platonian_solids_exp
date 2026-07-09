#!/bin/bash
# Submits every real-experiment job_script_*.sh under new_experiments/ via sbatch
# (everything except job_script_hp_*.sh, which is hp-tuning — see submit_all_hp_tuning.sh).
# SchNet is excluded -- run locally, not on the cluster.
#
# Usage:
#   ./submit_all_experiments.sh           # submit for real
#   ./submit_all_experiments.sh --dry-run # just list what would be submitted
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

SCRIPTS=(
    "fullerene_randomsplit/job_script_cin.sh"
    "fullerene_randomsplit/job_script_ct.sh"
    "fullerene_randomsplit/job_script_fullerenet.sh"
    "fullerene_randomsplit/job_script_gnn.sh"
    "fullerene/job_script_cin.sh"
    "fullerene/job_script_ct.sh"
    "fullerene/job_script_fullerenet.sh"
    "fullerene/job_script_gnn.sh"
    "lrgb/peptides_func/job_script_ct.sh"
    "lrgb/peptides_func/job_script_gnn.sh"
    "lrgb/peptides_struct/job_script_cin.sh"
    "lrgb/peptides_struct/job_script_ct.sh"
    "lrgb/peptides_struct/job_script_gnn.sh"
    "mol3d_fullerene/job_script_cin.sh"
    "mol3d_fullerene/job_script_ct.sh"
    "mol3d_fullerene/job_script_gnn.sh"
    "mol3d/job_script_cin.sh"
    "mol3d/job_script_ct.sh"
    "mol3d/job_script_gnn.sh"
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

echo "Total experiment job scripts: ${#SCRIPTS[@]}"
