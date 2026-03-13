#!/bin/bash -l
#SBATCH --account=CNHPC_1498509
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
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
# PIN_DIR = {RUNS_DIR}/{RUN_BASENAME}_{ID}/*pinocchio files*
#
# - save:
# OUT_DIR = {MAINOUT_DIR}/{RUN_BASENAME}_{ID}/*CPP dirs*
#
# N_START <= ID < N_END
# ============================================ #

RUN_BASENAME="${RUN_BASENAME:-model}"

RUNS_DIR="${RUNS_DIR:-$HOME/path/to/all/runs/}"
MAINOUT_DIR="${MAINOUT_DIR:-$HOME/path/to/all/runs/outputs}"

MODE="${MODE:-particles}"

# ============================================ #
# SBATCH
# ============================================ #

BASH_SCRIPT="run_cosmopost_painting.sh"

declare -A WHICH_NUM_SHELLS=(
	["ncms"]=29
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
echo -e "\033[32m[JOB]\033[0m RUN_BASENAME ....... = ${RUN_BASENAME}"
echo -e "\033[32m[JOB]\033[0m DATA_SET ........... = ${DATA_SET}"
echo -e "\033[32m[JOB]\033[0m STARTING ........... = ${N_START}"
echo -e "\033[32m[JOB]\033[0m ENDING ............. = ${N_END}"
echo -e "\033[32m[JOB]\033[0m MODE ............... = ${MODE}"
echo -e "\033[32m[JOB]\033[0m .................... " 
echo -e "\033[32m[JOB]\033[0m SCRIPT ............. = ${BASH_SCRIPT}"

for ((m=$N_START; m<$N_END; m++)); do

	RUN_MODEL="${RUN_BASENAME}_$m"
	PIN_DIR="${RUNS_DIR}/${RUN_MODEL}"
	OUT_DIR="${MAINOUT_DIR}/${RUN_MODEL}"
	PARAM_PATH="${PIN_DIR}/parameter_file_${RUN_MODEL}_sig8"
	
	EXPORTS=(
		PIN_DIR="$PIN_DIR"
		OUT_DIR="$OUT_DIR"
		RUN_MODEL="$RUN_MODEL"
		PARAM_PATH="$PARAM_PATH"
		MODE="$MODE"
		DATA_SET="$DATA_SET"
	)
	EXPORTS_STR=$(IFS=,; echo "${EXPORTS[*]}")

	sbatch \
		--job-name=painting_${RUN_MODEL} \
		--output=../logs/painting/${RUN_MODEL}/slurm-%x_%A_%a.out \
		--array=0-$NUM_SHELLS%$NUM_SHELLS \
		--export="$EXPORTS_STR" \
		"${BASH_SCRIPT}"
	
done

echo -e "\033[32m[JOB]\033[0m All done." 
