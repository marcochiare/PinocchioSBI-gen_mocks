#!/bin/bash
#SBATCH --account=IscrC_SBIxEuCG
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=zshells
#SBATCH --array=0-10%10
#SBATCH --output=../logs/zshells/slurm-%x_%a.out

# PLEASE, SBATCH THIS FILE FROM INSIDE jobs/
# (otherwise it will not work as intended)

# ====================== #
# MODULES & LIBRARIES
# ====================== #

module load python/3.10.8--gcc--8.5.0

source $HOME/dott/envs/gen_mocks/bin/activate

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

# Name prefix
PREFIX="${PREFIX:-plc_shell}"

# Whether to use or not the redshifts with RDS
USE_TRUEZ="${USE_TRUEZ:-0}"

# Read the mass calibrated PLC from a .fits file instead of the .out file
USE_FITS="${USE_FITS:-1}"

# Python script for splitting the PLC in shells
PY_SCRIPT="${PY_SCRIPT:-../scripts/plc_massshells_parser.py}"

# Combine base_name and directory (NO NEED TO CHANGE)
ID="$SLURM_ARRAY_TASK_ID"
RUN_NAME="${RUN_BASENAME}_${SLURM_ARRAY_TASK_ID}"
SIM_DIR="$RUNS_DIR/$RUN_NAME"

# How the parameter file is named in each run directory
PARAMFILE="${PARAMFILE:-parameter_file_$RUN_NAME}"

# How the file containing the redshift shells in named in each run directory
SHELLSFILE="${SHELLSFILE:-outputs_shells}"

# Name of the status file
STATUSNAMEFILE="${STATUSNAMEFILE:-status.txt}"
STATUSFILE="$RUNS_DIR/$STATUSNAMEFILE"

# ====================== #
# PRINTS
# ====================== #

echo -e "\033[32m[JOB]\033[0m ${SLURM_ARRAY_JOB_ID}_${ID}"
echo -e "\033[32m[JOB]\033[0m $(date +"%Y-%m-%d %H:%M:%S")"
echo -e "\033[32m[JOB]\033[0m RUNS_DIR ........... = ${RUNS_DIR}" 
echo -e "\033[32m[JOB]\033[0m RUN_NAME ........... = ${RUN_NAME}" 
echo -e "\033[32m[JOB]\033[0m PARAMFILE .......... = ${PARAMFILE}" 
echo -e "\033[32m[JOB]\033[0m SHELLSFILE ......... = ${SHELLSFILE}" 
echo -e "\033[32m[JOB]\033[0m PREFIX ............. = ${PREFIX}" 
echo -e "\033[32m[JOB]\033[0m USE_TRUEZ .......... = ${USE_TRUEZ}" 
echo -e "\033[32m[JOB]\033[0m USE_FITS FILE ...... = ${USE_FITS}" 
echo -e "\033[32m[JOB]\033[0m .................... " 
echo -e "\033[32m[JOB]\033[0m PYTHON ............. = $(which python)"
echo -e "\033[32m[JOB]\033[0m PYTHON VERSION ..... = $(python -V)"
echo -e "\033[32m[JOB]\033[0m SCRIPT ............. = ${PY_SCRIPT}"

# ====================== #
# CHECKS
# ====================== #

if ! [ -f "$STATUSFILE" ]; then
	echo -e "\033[31m[ERR]\033[0m Status file $STATUSFILE not found." 
	exit 1
fi

if ! [ -f "${SIM_DIR}/${PARAMFILE}" ]; then
	echo -e "\033[31m[ERR]\033[0m Param. file $PARAMFILE not found in $SIM_DIR."
	exit 1
fi

# Check that status in STATUSFILE is "done" or "zshells-FAILED" before starting
CURRENT_STATUS=$(sed -n "s/^$RUN_NAME\s\+//p" "$STATUSFILE")

if [[ ! "$CURRENT_STATUS" =~ ^(done|zshells-FAILED)$ ]]; then
	echo -e "\033[31m[ERR]\033[0m $RUN_NAME has an invalid status. Found $CURRENT_STATUS."
	exit 1
fi

echo -e "\033[32m[JOB $(date +"%H:%M:%S")]\033[0m Running $RUN_NAME"

# ====================== #
# ARGS
# ====================== #

ARGS=(
	--pin-dir "${SIM_DIR}"
	--param-file "${SIM_DIR}/${PARAMFILE}"
	--shells-file "${SIM_DIR}/${SHELLSFILE}"
	--out-prefix "${PREFIX}"
)

if [[ "${USE_TRUEZ}" == "1" ]]; then
	ARGS+=(--use-truez)
fi

if [[ "${USE_FITS}" == "1" ]]; then
	ARGS+=(--use-fits)
fi

# ====================== #
# RUN
# ====================== #

# Update the status in STATUSFILE with "splitting"
STATUS="splitting"
sed -i "s/^\($RUN_NAME\s\+\).*/\1$STATUS/" "$STATUSFILE"

python "${PY_SCRIPT}" "${ARGS[@]}" && STATUS="zshells-done" || STATUS="zshells-FAILED"

# Update the status in STATUSFILE based on the exit code
sed -i "s/^\($RUN_NAME\s\+\).*/\1$STATUS/" "$STATUSFILE"

if [[ "$STATUS" == "zshells-FAILED" ]]; then
	echo -e "\033[31m[ERR]\033[0m Splitting the PLC into shells has failed."
	exit 1
fi

echo -e "\033[32m[JOB $(date +"%H:%M:%S")]\033[0m All done."
exit 0
