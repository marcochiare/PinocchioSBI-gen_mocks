#!/bin/bash
#SBATCH --account=
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --job-name=run_pinocchio
#SBATCH --array=0-1%10
#SBATCH --output=../logs/runs/slurm-%x_%A_%a.out

# PLEASE, SBATCH THIS FILE FROM INSIDE jobs/
# (otherwise it will not work as intended)

# ====================== #
# MODULES
# ====================== #

module load openmpi/
module load gsl/
module load fftw/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/pfft/lib

# ====================== #
# PATHS & NAMES
# 
# Expected structure:
# {MAIN_DIR}/{RUN_BASENAME}_{ID}/{PARAMFILE}
#
# ====================== #

# Directory containing all setupped runs
MAIN_DIR="${MAIN_DIR:-$WORK}"

# Base name for the Pinocchio run --> (base_name)_(id)
RUN_BASENAME="${RUN_BASENAME:-model}"

# Path to Pinocchio executable
EXEC="${EXEC:-pinocchio.x}"

# Combine base_name and directory (NO NEED TO CHANGE)
ID="$SLURM_ARRAY_TASK_ID"
RUN_NAME="${RUN_BASENAME}_${SLURM_ARRAY_TASK_ID}"
SIM_DIR="$MAIN_DIR/$RUN_NAME"

# How the parameter file is named in each run directory
PARAMFILE="${PARAMFILE:-parameter_file_$RUN_NAME}"

# ====================== #
# PRINTS
# ====================== #

echo -e "\033[32m[JOB]\033[0m ${SLURM_ARRAY_JOB_ID}_${ID}"
echo -e "\033[32m[JOB]\033[0m $(date +"%Y-%m-%d %H:%M:%S")"
echo -e "\033[32m[JOB]\033[0m MAIN_DIR ........... = ${MAIN_DIR}" 
echo -e "\033[32m[JOB]\033[0m RUN_NAME ........... = ${RUN_NAME}" 
echo -e "\033[32m[JOB]\033[0m PARAMFILE .......... = ${PARAMFILE}" 
echo -e "\033[32m[JOB]\033[0m .................... " 
echo -e "\033[32m[JOB]\033[0m PINOCCHIO .......... = ${EXEC}" 
echo -e "\033[32m[JOB]\033[0m NODES .............. = ${SLURM_JOB_NUM_NODES}"
echo -e "\033[32m[JOB]\033[0m NTASKS-per-NODE .... = ${SLURM_NTASKS_PER_NODE}"
echo -e "\033[32m[JOB]\033[0m CPUS-per-TASK ...... = ${SLURM_CPUS_PER_TASK}"

# ====================== #
# CHECKS
# ====================== #

cd $SIM_DIR || { echo -e "\033[31m[ERR]\033[0m Directory $SIM_DIR not found."; exit 1; }

if ! [ -f "$PARAMFILE" ]; then
	echo -e "\033[31m[ERR]\033[0m Param. file $PARAMFILE not found in $SIM_DIR."
	exit 1
fi

echo -e "\033[32m[JOB $(date +"%H:%M:%S")]\033[0m Running simulation $RUN_NAME"

# ====================== #
# RUN
# ====================== #

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
MPI_PROCS="$SLURM_NTASKS_PER_NODE"
LOG_FILE="pinocchio_$RUN_NAME.log"

mpirun -n $MPI_PROCS $EXEC $PARAM > $LOG_FILE

echo -e "\033[32m[JOB $(date +"%H:%M:%S")]\033[0m Run finished. Log saved to $LOG_FILE"
