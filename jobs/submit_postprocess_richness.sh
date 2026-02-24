#!/bin/bash -l
#SBATCH --account=iscrc_graphmls
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=submit_postprocess_richness
#SBATCH --output=../logs/slurm-%x_%j.out

# PLEASE, SBATCH THIS FILE FROM INSIDE jobs/
# (otherwise it will not work as intended)

# ============================================ #
# This script is used to sbatch multiple jobs
# that post-process the PINOCCHIO runs.
# ============================================ #

# ============================================ #
# NUMBER OF PINOCCHIO TO POST-PROCESS
#
# You must have painted the catalogues with
# galaxies, so send this script after the other
# ============================================ #

N_START=0
N_END=10

# ============================================ #
# PATHS
#
# Expected structure:
# - read parameter file
# PIN_DIR = {RUNS_DIR}/{MODEL_BASE}_{ID}/*pinocchio files*
#
# - read the post-processed files and save (default):
# PP_DIR = {MAINPP_DIR}/{MODEL_BASE}_{ID}/*CPP painting dirs*
#
# (save alt. dir, optionally)
# OUT_DIR = {MAINOUT_DIR}/{MODEL_BASE}_{ID}/*CPP richness dirs*
#
# N_START <= ID < N_END
# ============================================ #

MODEL_BASE="${MODEL_BASE:-model}"

RUNS_DIR="${RUNS_DIR:-$HOME/path/to/all/runs/}"
MAINPP_DIR="${MAINPP_DIR:-$HOME/path/to/all/painting/outputs}"
MAINOUT_DIR="${MAINOUT_DIR:-$PP_DIR}"

# ============================================ #
# SBATCH
# ============================================ #

BASH_SCRIPT="run_cosmopost_richness.sh"

echo -e "\033[32m[JOB]\033[0m ${SLURM_JOB_ID}"
echo -e "\033[32m[JOB]\033[0m $(date +"%Y-%m-%d %H:%M:%S")"
echo -e "\033[32m[JOB]\033[0m RUNS_DIR ........... = ${RUNS_DIR}"
echo -e "\033[32m[JOB]\033[0m MAINPP_DIR ......... = ${MAINPP_DIR}"
echo -e "\033[32m[JOB]\033[0m MAINOUT_DIR ........ = ${MAINOUT_DIR}"
echo -e "\033[32m[JOB]\033[0m MODEL_BASE ......... = ${MODEL_BASE}"
echo -e "\033[32m[JOB]\033[0m STARTING ........... = ${N_START}"
echo -e "\033[32m[JOB]\033[0m ENDING ............. = ${N_END}"
echo -e "\033[32m[JOB]\033[0m .................... " 
echo -e "\033[32m[JOB]\033[0m SCRIPT ............. = ${BASH_SCRIPT}"

for ((m=$N_START; m<$N_END; m++)); do

	RUN_MODEL="${MODEL_BASE}_$m"
	PIN_DIR="${RUNS_DIR}/${RUN_MODEL}"
	PP_DIR="${MAINPP_DIR}/${RUN_MODEL}"
	OUT_DIR="${MAINOUT_DIR}/${RUN_MODEL}"
	PARAM_PATH="${PIN_DIR}/parameter_file_${RUN_MODEL}"
	
	sbatch \
		--job-name=richness_${RUN_MODEL} \
		--export=PIN_DIR="$PIN_DIR",OUT_DIR="$OUT_DIR",PARAM_PATH="$PARAM_PATH" \
		"$BASH_SCRIPT"
	
done

echo -e "\033[32m[JOB]\033[0m All done." 
