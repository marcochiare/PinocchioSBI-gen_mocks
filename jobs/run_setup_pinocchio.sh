#!/bin/bash
#SBATCH --account=CNHPC_1498509
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

source $HOME/dott/envs/cosmopostprocess/bin/activate

# ====================== #
# PATHS & NAMES
# ====================== #

# Where to setup the runs
RUNS_DIR="${RUNS_DIR:-$HOME/dott/runs}"

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
# A box catalogue is saved for each one of these redshift, and, if the 
# mass shells are enabled, an healpix map.
Z_SNAP=(
	1.0
	0.1
	0.0
)

# Redshift shells to use in the painting process. This should correspond to
# Z_SNAP if the mass shells are produced. BE CAREFUL! Also these values should
# correspond to the data set used in the painting
Z_SHELLS=(
	2.0 0.9
	1.8 0.8
	1.7 0.7
	1.6 0.6
	1.5 0.5
	1.4 0.4
	1.3 0.3
	1.2 0.2
	1.1 0.1
	1.0 0.0
)

# LPT IC snapshot
Z_IC="50."

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
	MassMapNSIDE='DISABLE'
	PLCProvideConeData='DISABLE'
	PLCCenter='DISABLE'
	PLCAxis='DISABLE'
	CatalogInAscii='DISABLE'
	MinHaloMass=10
	BoundaryLayerFactor=2.5
	MaxMem=8500
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
echo -e "\033[32m[JOB]\033[0m RUNS_DIR ........... = ${RUNS_DIR}"
echo -e "\033[32m[JOB]\033[0m RUN_NAME ........... = ${RUN_BASENAME}"
echo -e "\033[32m[JOB]\033[0m TOT_RUNS ........... = ${TOT_RUNS}"
echo -e "\033[32m[JOB]\033[0m COSMO_FILE ......... = ${COSMO_FILE}"
echo -e "\033[32m[JOB]\033[0m PARAMS ............. = ${PARAMS[@]}"
echo -e "\033[32m[JOB]\033[0m SNAPSHOTS .......... = ${Z_SNAP[@]}"
echo -e "\033[32m[JOB]\033[0m SHELLS ............. = ${Z_SHELLS[@]}"
echo -e "\033[32m[JOB]\033[0m LPT SNAPSHOT IC .... = ${Z_IC}"
echo -e "\033[32m[JOB]\033[0m OPT. SETUP ......... = ${SETUP[@]}"
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
	--cosmo-file "${COSMO_FILE}"
	--total-runs "${TOT_RUNS}"
	--z-out "${Z_SNAP[@]}"
	--z-out-shells "${Z_SHELLS[@]}"
	--z-IC "${Z_IC}"
	--params "${PARAMS[@]}"
	--setup-args "${SETUP[@]}"
)

python "${PY_SCRIPT}" "${ARGS[@]}"

echo -e "\033[32m[JOB]\033[0m Job finished."
