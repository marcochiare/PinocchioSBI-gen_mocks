#!/bin/bash
#SBATCH --account=CMPNS_units
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --time=1:30:00
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --job-name=run_pinocchio
#SBATCH --array=0-10%2
#SBATCH --output=../logs/runs/$x_%A/slurm-%x_%a.out

# PLEASE, SBATCH THIS FILE FROM INSIDE jobs/
# (otherwise it will not work as intended)

# ====================== #
# MODULES & LIBRARIES
# ====================== #

module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3

LIB=/leonardo_scratch/fast/CNHPC_1498509/lib/nvhpc-23.11
LIB_Mass_shell=/leonardo_scratch/large/userexternal/tbatalha/Pinocchio/dep

HEALPIX_LIB="${LIB_Mass_shell}/Healpix_3.83/lib"
CFITSIO_LIB="${LIB_Mass_shell}/lib"
FTTW_LIB="${LIB}/fftw/fftw-3.3.10/lib"
GSL_LIB="${LIB}/gsl/gsl-2.7.1/lib"
PFFT_LIB="${LIB}/pfft/pfft/lib"

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${FTTW_LIB}:${GSL_LIB}:${PFFT_LIB}:${PMT_LIB}:${HEALPIX_LIB}:${CFITSIO_LIB}

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

# Path to Pinocchio executable
EXEC="${EXEC:-pinocchio.x}"

# Combine base_name and directory (NO NEED TO CHANGE)
ID="$SLURM_ARRAY_TASK_ID"
RUN_NAME="${RUN_BASENAME}_${SLURM_ARRAY_TASK_ID}"
SIM_DIR="$RUNS_DIR/$RUN_NAME"

# How the parameter file is named in each run directory
PARAMFILE="${PARAMFILE:-parameter_file_$RUN_NAME}"

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
echo -e "\033[32m[JOB]\033[0m .................... " 
echo -e "\033[32m[JOB]\033[0m PINOCCHIO .......... = ${EXEC}" 
echo -e "\033[32m[JOB]\033[0m SYSTEM ............. = ${SLURM_PARTITION}" 
echo -e "\033[32m[JOB]\033[0m NODES .............. = ${SLURM_JOB_NUM_NODES}"
echo -e "\033[32m[JOB]\033[0m TOTAL TASKS ........ = ${SLURM_NTASKS}"
echo -e "\033[32m[JOB]\033[0m NTASKS-per-NODE .... = ${SLURM_NTASKS_PER_NODE}"
echo -e "\033[32m[JOB]\033[0m CPUS-per-TASK ...... = ${SLURM_CPUS_PER_TASK}"
echo -e "\033[32m[JOB]\033[0m NODELIST ........... = ${SLURM_NODELIST}"

# ====================== #
# CHECKS
# ====================== #

if ! [ -f "$STATUSFILE" ]; then
	echo -e "\033[31m[ERR]\033[0m Status file $STATUSFILE not found." 
	exit 1
fi

cd $SIM_DIR || { echo -e "\033[31m[ERR]\033[0m Directory $SIM_DIR not found."; exit 1; }

if ! [ -f "$PARAMFILE" ]; then
	echo -e "\033[31m[ERR]\033[0m Param. file $PARAMFILE not found in $SIM_DIR."
	exit 1
fi

echo -e "\033[32m[JOB $(date +"%H:%M:%S")]\033[0m Running $RUN_NAME"

# ====================== #
# RUN
# ====================== #

# the following exports are not strictly fundamental but may be necessary on some clusters
export OMP_TARGET_OFFLOAD=mandatory
export OMP_PROC_BIND=true
export OMP_WAIT_POLICY=ACTIVE
export OMP_PLACES=cores

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MPI_PROCS="$SLURM_NTASKS --map-by ppr:$SLURM_NTASKS_PER_NODE:node:pe=$OMP_NUM_THREADS"
LOG_FILE="pinocchio_$RUN_NAME.log"

# Update the status in STATUSFILE with "running"
STATUS="running"
sed -i "s/^\($RUN_NAME\s\+\).*/\1$STATUS/" "$STATUSFILE"

echo -e "\033[32m[JOB $(date +"%H:%M:%S")]\033[0m mpirun -n $MPI_PROCS $EXEC $PARAMFILE > $LOG_FILE"
mpirun -n $MPI_PROCS $EXEC $PARAMFILE > $LOG_FILE && STATUS="done" || STATUS="FAILED"

# Update the status in STATUSFILE based on the exit code
sed -i "s/^\($RUN_NAME\s\+\).*/\1$STATUS/" "$STATUSFILE"

if [ "$STATUS" = "FAILED" ]; then
	echo -e "\033[31m[ERR]\033[0m Run failed. Log saved to $LOG_FILE"
	exit 1
fi

echo -e "\033[32m[JOB $(date +"%H:%M:%S")]\033[0m Run finished. Log saved to $LOG_FILE"
exit 0
