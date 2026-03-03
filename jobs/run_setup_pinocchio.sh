#!/bin/bash
#SBATCH --account=
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=setup_pinocchio
#SBATCH --output=../logs/slurm-%x_%j.out

# PLEASE, SBATCH THIS FILE FROM INSIDE jobs/
# (otherwise it will not work as intended)

# print python unbuffered to the logs
export PYTHONUNBUFFERED=1

# ====================== #
# PYTHON ENVIRONMENT
# ====================== #

module load python/3.10.8--gcc--8.5.0

source $HOME/dott/env/gen_mocks/bin/activate

# ====================== #
# PATHS & NAMES
# ====================== #

# Where to setup the runs
MAIN_DIR="${MAIN_DIR:-$HOME/dott/runs}"

# python script to run
PY_SCRIPT="${PY_SCRIPT:-../scripts/setup_runs_parser.py}"

# Base name for the Pinocchio run --> (base_name)_(id)
RUN_BASENAME="${RUN_BASENAME:-model}"

# Total number of runs to setup
TOT_RUNS="${TOT_RUNS:-10}"

# ====================== #
# PARAMETERS from SOBOL
# ====================== #

# Cosmo Param file
COSMO_FILE="${COSMO_FILE:-../SobolSeq/models_parameters_3dim.txt}"

# Parameter list to use (from COSMO_FILE)
PARAMS=(
	"Omega_m"
	"sigma_8"
	"h"
)

# ====================== #
# REDSHIFT SNAPSHOTS
# ====================== #

# Always include 0. or whatever ending redshift for the simulation
# (values are sorted anyways)
Z_SNAP=(
	2.000000000000
	1.907434262995
	1.723188618705
	1.554382912656
	1.399160301249
	1.255918543942
	1.000000000003
	0.942992679411
	0.886451684155
	0.831760033291
	0.778812007111
	0.727508223193
	0.677755169340
	0.629464776814
	0.582554030021
	0.536944609231
	0.492562563187
	0.449338008940
	0.407204856351
	0.366100555129
	0.325965862358
	0.286744628730
	0.248383601907
	0.210832245519
	0.174042572504
	0.137968991589
	0.102568165866
	0.067798882457
	0.033621932408
	0.000000000000
)

# ====================== #
# PINOCCHIO SPECS
# ====================== #

# Add pairs of param_key=value to pass to the param. file
# Use the names expected in Pinocchio!
# DO NOT ADD SPACES!!
SETUP=(
	BoxSize=3870.
	GridSize=2160
	OmegaBaryon=0.049
	PrimordialIndex=0.96
	StartingzForPLC=2.0
	PLCAperture=70.
	MassMapNSIDE=2048
	PLCProvideConeData='DISABLE'
	PLCCenter='DISABLE'
	PLCAxis='DISABLE'
	CatalogInAscii='DISABLE'
	MinHaloMass=10
	BoundaryLayerFactor=2.5
	MaxMem=31000
	MaxMemPerParticle=350
	PredPeakFactor=1.0
	seed=115
)

# NB: do not use "RandomSeed", only "seed"
# The unique value "seed*id" will be assigned to each run

# ====================== #
# PRINTS
# ====================== #

echo -e "\033[32m[JOB]\033[0m ${SLURM_JOB_ID}"
echo -e "\033[32m[JOB]\033[0m $(date +"%Y-%m-%d %H:%M:%S")"
echo -e "\033[32m[JOB]\033[0m MAIN_DIR ........... = ${MAIN_DIR}"
echo -e "\033[32m[JOB]\033[0m RUN_NAME ........... = ${RUN_BASENAME}"
echo -e "\033[32m[JOB]\033[0m TOT_RUNS ........... = ${TOT_RUNS}"
echo -e "\033[32m[JOB]\033[0m COSMO_FILE ......... = ${COSMO_FILE}"
echo -e "\033[32m[JOB]\033[0m PARAMS ............. = ${PARAMS[@]}"
echo -e "\033[32m[JOB]\033[0m SNAPSHOTS .......... = ${Z_SNAP[@]}"
echo -e "\033[32m[JOB]\033[0m OPT. SETUP ......... = ${SETUP[@]}"
echo -e "\033[32m[JOB]\033[0m .................... " 
echo -e "\033[32m[JOB]\033[0m PYTHON ............. = $(which python)"
echo -e "\033[32m[JOB]\033[0m PYTHON VERSION ..... = $(python -V)"
echo -e "\033[32m[JOB]\033[0m SCRIPT ............. = ${PY_SCRIPT}"

# ====================== #
# ARGS
# ====================== #

ARGS=(
	--main-dir "${MAIN_DIR}"	
	--base-name "${RUN_BASENAME}"
	--cosmo-file "${COSMO_FILE}"
	--total-runs "${TOT_RUNS}"
	--z-out "${Z_SNAP[@]}"
	--params "${PARAMS[@]}"
	--setup-args "${SETUP[@]}"
)

python "${PY_SCRIPT}" "${ARGS[@]}"

echo -e "\033[32m[JOB]\033[0m Job finished."
