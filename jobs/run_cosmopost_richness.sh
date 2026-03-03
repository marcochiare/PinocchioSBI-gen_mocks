#!/bin/bash -l
#SBATCH --account=iscrc_graphmls
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --job-name=richness_merged
#SBATCH --output=../logs/richness/slurm-%x_%j.out

set -euo pipefail

# ----------------------------
# Threading: keep libs single-threaded inside each multiprocessing worker
# ----------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ----------------------------
# MKL / conda activation fix
# ----------------------------
export MKL_INTERFACE_LAYER=GNU
export MKL_THREADING_LAYER=GNU

# Unbuffered python so progress prints into slurm logs
export PYTHONUNBUFFERED=1

# ----------------------------
# Environment
# ----------------------------
module load python/3.10.8--gcc--8.5.0
source $HOME/dott/envs/cosmopostprocess/bin/activate

# ----------------------------
# Paths
# ----------------------------

# The directory containing the outputs of the painting process (could coincide or not with the PINOCCHIO dir)
PIN_DIR="${PIN_DIR:-$WORK/mchiaren/}"

# Where to save the products. By default, these are saved in the PIN_DIR
OUT_DIR="${OUT_DIR:-${PIN_DIR}}"

# The PINOCCHIO parameter file for the run
PARAM_PATH="${PARAM_PATH:-$PIN_DIR/parameter_file}"

# Richness script from COSMOPOSTPROCESS
PY_SCRIPT="${PY_SCRIPT:-../scripts/cosmopostprocess_richness.py}"

HALO_ROOT="${HALO_ROOT:-${PIN_DIR}}"
HALO_PATTERN="${HALO_PATTERN:-plc_shell_thetaphizmass_z*.npz}"   # auto-skips *_tidal_proxy.npz in the .py
GALAXY_ROOT="${GALAXY_ROOT:-${PIN_DIR}/outputs_particle}"

OUT_DIR_FULL="${OUT_DIR}/richness_global"
mkdir -p "${OUT_DIR_FULL}"

OUT_HALO_NPZ="${OUT_HALO_NPZ:-${OUT_DIR_FULL}/halo_richness_merged_zmax2.0.npz}"
OUT_GAL_NPZ="${OUT_GAL_NPZ:-${OUT_DIR_FULL}/galaxies_merged_zmax2.0.npz}"

Z_MAX_SHELL="${Z_MAX_SHELL:-2.0}"

# ----------------------------
# Resources / model
# ----------------------------
PROCS="${PROCS:-${SLURM_CPUS_PER_TASK}}"   # multiprocessing pool size
LAMBDA_NUM_WORKERS="${LAMBDA_NUM_WORKERS:-4}"  # inner workers in lambda_observed per process

PMEM_MODEL_PATH="${PMEM_MODEL_PATH:-../Pmem_models/Pmem_model_opt_370deg2_cpz_pzwav_bkgf.pt}"

# Major-contributor threshold (pairs recorded in galaxy npz)
PMEM_MAJOR_THRESHOLD="${PMEM_MAJOR_THRESHOLD:-0.01}"

# Mass threshold for computing R200/lambda_obs/f_bkg (same units as Mass)
M_THRESHOLD="${M_THRESHOLD:-1e13}"

# Optional: disable tqdm bars (0/1)
NO_PROGRESS="${NO_PROGRESS:-0}"

echo -e "\033[32m[Job]\033[0m PIN_DIR               = ${PIN_DIR}"
echo -e "\033[32m[Job]\033[0m PARAM_PATH            = ${PARAM_PATH}"
echo -e "\033[32m[Job]\033[0m PY_SCRIPT             = ${PY_SCRIPT}"
echo -e "\033[32m[Job]\033[0m HALO_ROOT             = ${HALO_ROOT}"
echo -e "\033[32m[Job]\033[0m HALO_PATTERN          = ${HALO_PATTERN}"
echo -e "\033[32m[Job]\033[0m GALAXY_ROOT           = ${GALAXY_ROOT}"
echo -e "\033[32m[Job]\033[0m OUT_HALO_NPZ          = ${OUT_HALO_NPZ}"
echo -e "\033[32m[Job]\033[0m OUT_GAL_NPZ           = ${OUT_GAL_NPZ}"
echo -e "\033[32m[Job]\033[0m Z_MAX_SHELL           = ${Z_MAX_SHELL}"
echo -e "\033[32m[Job]\033[0m M_THRESHOLD           = ${M_THRESHOLD}"
echo -e "\033[32m[Job]\033[0m PROCS                 = ${PROCS}"
echo -e "\033[32m[Job]\033[0m LAMBDA_NUM_WORKERS    = ${LAMBDA_NUM_WORKERS}"
echo -e "\033[32m[Job]\033[0m PMEM_MODEL_PATH       = ${PMEM_MODEL_PATH}"
echo -e "\033[32m[Job]\033[0m PMEM_MAJOR_THRESHOLD  = ${PMEM_MAJOR_THRESHOLD}"
echo -e "\033[32m[Job]\033[0m NO_PROGRESS           = ${NO_PROGRESS}"
echo -e "\033[32m[Job]\033[0m PYTHON				  = $(which python)"
echo -e "\033[32m[Job]\033[0m PYTHON VERSION        = $(python -V)"
echo -e "\033[32m[Job]\033[0m Note: f_bkg uses fixed annulus 3-5 cMpc/h inside compute_richness_merged.py"

# ----------------------------
# Run global richness computation
# ----------------------------
ARGS=(
  --halo-root "${HALO_ROOT}"
  --halo-pattern "${HALO_PATTERN}"
  --galaxy-root "${GALAXY_ROOT}"
  --paramfile-path "${PARAM_PATH}"
  --out-halo-npz "${OUT_HALO_NPZ}"
  --out-galaxy-npz "${OUT_GAL_NPZ}"
  --pmem-model-path "${PMEM_MODEL_PATH}"
  --lambda-num-workers "${LAMBDA_NUM_WORKERS}"
  --procs "${PROCS}"
  --z-max-shell "${Z_MAX_SHELL}"
  --m-threshold "${M_THRESHOLD}"
  --pmem-major-threshold "${PMEM_MAJOR_THRESHOLD}"
)

if [[ "${NO_PROGRESS}" == "1" ]]; then
  ARGS+=(--no-progress)
fi

srun python -u "${PY_SCRIPT}" "${ARGS[@]}"


#echo -e "\033[32m[JOB]\033[0m Removing galaxy cat"

echo -e "\033[32m[JOB]\033[0m finished"
