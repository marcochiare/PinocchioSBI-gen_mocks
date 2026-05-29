#!/bin/bash
#SBATCH --account=IscrC_SBIxEuCG
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=mass_calibration
#SBATCH --output=../logs/calibration/slurm-%x_%j.out

# PLEASE, SBATCH THIS FILE FROM INSIDE jobs/
# (otherwise it will not work as intended)

# ====================== #
# MODULES & LIBRARIES
# ====================== #

module load python/3.10.8--gcc--8.5.0

source $HOME/dott/envs/cosmopostprocess/bin/activate

set -euo pipefail

# ====================== #
# PATHS & NAMES
# 
# Expected structure:
# {RUNS_DIR}/{RUN_BASENAME}_{ID}/{PARAMFILE}
# {RUNS_DIR}/{STATUSNAMEFILE}
#
# ====================== #

# Directory containing all setup runs
RUNS_DIR="${RUNS_DIR:-$WORK}"

# Base name for the Pinocchio run --> (base_name)_(id)
RUN_BASENAME="${RUN_BASENAME:-model}"

# Python script for splitting the PLC in shells
PY_SCRIPT="${PY_SCRIPT:-../scripts/calibrate_sbi.py}"

N_START=1
N_END=2

# ====================== #
# PRINTS
# ====================== #

echo -e "\033[32m[JOB]\033[0m ${SLURM_JOB_ID}"
echo -e "\033[32m[JOB]\033[0m $(date +"%Y-%m-%d %H:%M:%S")"
echo -e "\033[32m[JOB]\033[0m RUNS_DIR ........... = ${RUNS_DIR}" 
echo -e "\033[32m[JOB]\033[0m RUN_BASENAME ....... = ${RUN_BASENAME}" 
echo -e "\033[32m[JOB]\033[0m N_START ............ = ${N_START}"
echo -e "\033[32m[JOB]\033[0m N_END .............. = ${N_END}"
echo -e "\033[32m[JOB]\033[0m .................... " 
echo -e "\033[32m[JOB]\033[0m PYTHON ............. = $(which python)"
echo -e "\033[32m[JOB]\033[0m PYTHON VERSION ..... = $(python -V)"
echo -e "\033[32m[JOB]\033[0m SCRIPT ............. = ${PY_SCRIPT}"

# ====================== #
# ARGS
# ====================== #

ARGS=(
	--main-dir "${RUNS_DIR}"
	--base-name "${RUN_BASENAME}"
	--total-runs "${N_END}"
	--start-run "${N_START}"
)

# ====================== #
# RUN
# ====================== #

python "${PY_SCRIPT}" "${ARGS[@]}"

echo -e "\033[32m[JOB $(date +"%H:%M:%S")]\033[0m All done."
exit 0
