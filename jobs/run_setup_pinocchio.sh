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
	2.041
	1.900
	1.800
	1.700
	1.600
	1.500
	1.400
	1.270
	1.150
	1.019
	0.900
	0.796
	0.650
	0.506
	0.192
	0.000
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
	StartingzForPLC=2.041
	PLCAperture=70.
	MassMapNSIDE=2048
	PLCProvideConeData='DISABLE'
	PLCCenter='DISABLE'
	PLCAxis='DISABLE'
	CatalogInAscii='DISABLE'
	#NumFiles=4 # yes??
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
