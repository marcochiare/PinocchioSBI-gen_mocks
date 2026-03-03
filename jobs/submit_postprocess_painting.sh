#!/bin/bash -l
#SBATCH --account=iscrc_graphmls
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=submit_postprocess_painting
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
# ncms: NewClusterMocksSobol
# elb: EuclidLargeBox
# ============================================ #

DATA_SET="ncms"
N_START=0
N_END=10

# ============================================ #
# PATHS
#
# Expected structure:
# - read:
# PIN_DIR = {RUNS_DIR}/{MODEL_BASE}_{ID}/*pinocchio files*
#
# - save:
# OUT_DIR = {MAINOUT_DIR}/{MODEL_BASE}_{ID}/*CPP dirs*
#
# N_START <= ID < N_END
# ============================================ #

MODEL_BASE="${MODEL_BASE:-model}"

RUNS_DIR="${RUNS_DIR:-$HOME/path/to/all/runs/}"
MAINOUT_DIR="${MAINOUT_DIR:-$HOME/path/to/all/runs/outputs}"

MODE="${MODE:-particles}"

# ============================================ #
# SBATCH
# ============================================ #

BASH_SCRIPT="run_cosmopost_painting.sh"

declare -A WHICH_NUM_SHELLS=(
	["ncms"]=30
	["elb"]=24
)
NUM_SHELLS="${WHICH_NUM_SHELLS[$DATA_SET]}"

if [[ -z "$NUM_SHELLS" ]]; then
    echo "Wrong DATA_SET value: $DATA_SET"
    exit 1
fi

echo -e "\033[32m[JOB]\033[0m ${SLURM_JOB_ID}"
echo -e "\033[32m[JOB]\033[0m $(date +"%Y-%m-%d %H:%M:%S")"
echo -e "\033[32m[JOB]\033[0m RUNS_DIR ........... = ${RUNS_DIR}"
echo -e "\033[32m[JOB]\033[0m MAINOUT_DIR ........ = ${MAINOUT_DIR}"
echo -e "\033[32m[JOB]\033[0m MODEL_BASE ......... = ${MODEL_BASE}"
echo -e "\033[32m[JOB]\033[0m DATA_SET ........... = ${DATA_SET}"
echo -e "\033[32m[JOB]\033[0m STARTING ........... = ${N_START}"
echo -e "\033[32m[JOB]\033[0m ENDING ............. = ${N_END}"
echo -e "\033[32m[JOB]\033[0m MODE ............... = ${MODE}"
echo -e "\033[32m[JOB]\033[0m .................... " 
echo -e "\033[32m[JOB]\033[0m SCRIPT ............. = ${BASH_SCRIPT}"

for ((m=$N_START; m<$N_END; m++)); do

	RUN_MODEL="${MODEL_BASE}_$m"
	PIN_DIR="${RUNS_DIR}/${RUN_MODEL}"
	OUT_DIR="${MAINOUT_DIR}/${RUN_MODEL}"
	PARAM_PATH="${PIN_DIR}/parameter_file_${RUN_MODEL}"
	
	sbatch \
		--job-name=painting_${RUN_MODEL} \
		--array=0-$NUM_SHELLS%$NUM_SHELLS \
		--export=PIN_DIR="$PIN_DIR",OUT_DIR="$OUT_DIR",RUN_MODEL="$RUN_MODEL",PARAM_PATH="$PARAM_PATH",MODE="$MODE",DATA_SET="$DATA_SET" \
		"${BASH_SCRIPT}"
	
done

echo -e "\033[32m[JOB]\033[0m All done." 
