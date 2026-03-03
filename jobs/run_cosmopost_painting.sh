#!/bin/bash -l
#SBATCH --account=iscrc_graphmls
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --job-name=pin_shells
#SBATCH --array=0-24%24
#SBATCH --output=../logs/painting/slurm-%x_%A_%a.out

# PLEASE, SBATCH THIS FILE FROM INSIDE jobs/
# (otherwise it will not work as intended)

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
# MODE SWITCH (easy):
#   sbatch --export=MODE=map       run_pinocchio_module1.sbatch
#   sbatch --export=MODE=particles run_pinocchio_module1.sbatch
# If MODE is unset, defaults to map.
#
# Optional high-res halo painting (MAP mode only):
#   sbatch --export=MODE=map,HALO_NSIDE=4096 run_pinocchio_module1.sbatch
# ----------------------------
MODE="${MODE:-map}"                # "map" or "particles"
DISABLE_BARY="${DISABLE_BARY:-0}"  # 1 => pass --disable-baryonification (map mode only)
HALO_NSIDE="${HALO_NSIDE:-}"       # if non-empty, passed as --halo-nside (map mode only)

# ----------------------------
# Paths
# ----------------------------

# The PINOCCHIO dir, containing the mass shells etc.
PIN_DIR="${PIN_DIR:-$WORK/mchiaren/}"

# Where to save the products. By default, these are saved in the PIN_DIR
OUT_DIR="${OUT_DIR:-${PIN_DIR}}"

# Painting script from COSMOPOSTPROCESS
PY_SCRIPT="${PY_SCRIPT:-../scripts/cosmopostprocess_painting_module1.py}"

# PINOCCHIO RunFlag
RUN_MODEL="${RUN_MODEL:-model}"

# The PINOCCHIO parameter file for the run
PARAM_PATH="${PARAM_PATH:-$PIN_DIR/parameter_file}"

# ----------------------------
# Shell catalogue list (index 0 = z0.000_0.100 ... index 24 = z3.800_4.000)
# ----------------------------

# Which shell catalogue list to use
DATA_SET="${DATA_SET:-elb}"

# EuclidLargeMocks:
if [[ "$DATA_SET" == "elb" ]]; then
	NPZ_NAMES=(
	  "plc_shell_thetaphizmass_z0.000_0.100.npz"  "plc_shell_thetaphizmass_z0.100_0.200.npz"  "plc_shell_thetaphizmass_z0.200_0.300.npz"  "plc_shell_thetaphizmass_z0.300_0.400.npz"  "plc_shell_thetaphizmass_z0.400_0.500.npz"
	  "plc_shell_thetaphizmass_z0.500_0.600.npz"  "plc_shell_thetaphizmass_z0.600_0.700.npz"  "plc_shell_thetaphizmass_z0.700_0.800.npz"  "plc_shell_thetaphizmass_z0.800_0.900.npz"  "plc_shell_thetaphizmass_z0.900_1.000.npz"
	  "plc_shell_thetaphizmass_z1.000_1.200.npz"  "plc_shell_thetaphizmass_z1.200_1.400.npz"  "plc_shell_thetaphizmass_z1.400_1.600.npz"  "plc_shell_thetaphizmass_z1.600_1.800.npz"  "plc_shell_thetaphizmass_z1.800_2.000.npz"
	  "plc_shell_thetaphizmass_z2.000_2.200.npz"  "plc_shell_thetaphizmass_z2.200_2.400.npz"  "plc_shell_thetaphizmass_z2.400_2.600.npz"  "plc_shell_thetaphizmass_z2.600_2.800.npz"  "plc_shell_thetaphizmass_z2.800_3.000.npz"
	  "plc_shell_thetaphizmass_z3.000_3.200.npz"  "plc_shell_thetaphizmass_z3.200_3.400.npz"  "plc_shell_thetaphizmass_z3.400_3.600.npz"  "plc_shell_thetaphizmass_z3.600_3.800.npz"  "plc_shell_thetaphizmass_z3.800_4.000.npz"
	)

# NewClusterMocksSobol:
elif [[ "$DATA_SET" == "ncms" ]]; then
	NPZ_NAMES=(
	  "plc_shell_thetaphizmass_z0.000_0.100.npz"  "plc_shell_thetaphizmass_z0.100_0.200.npz"  "plc_shell_thetaphizmass_z0.200_0.300.npz"  "plc_shell_thetaphizmass_z0.300_0.400.npz"  "plc_shell_thetaphizmass_z0.400_0.500.npz"
	  "plc_shell_thetaphizmass_z0.500_0.600.npz"  "plc_shell_thetaphizmass_z0.600_0.700.npz"  "plc_shell_thetaphizmass_z0.700_0.800.npz"  "plc_shell_thetaphizmass_z0.800_0.900.npz"  "plc_shell_thetaphizmass_z0.900_1.000.npz"
	  "plc_shell_thetaphizmass_z1.000_1.200.npz"  "plc_shell_thetaphizmass_z1.200_1.400.npz"  "plc_shell_thetaphizmass_z1.400_1.600.npz"  "plc_shell_thetaphizmass_z1.600_1.800.npz"  "plc_shell_thetaphizmass_z1.800_2.000.npz"
	)
else
	echo "[ERROR] $DATA_SET is not a valid data set type"
	exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
NPZ_NAME="${NPZ_NAMES[$TASK_ID]}"
NPZ_FILE="${PIN_DIR}/${NPZ_NAME}"

# Map segment index is descending:
# seg024 -> z0.000_0.100 and seg000 -> z3.800_4.000
SEG=$((${#NPZ_NAMES[@]} - TASK_ID - 1))
SEG3=$(printf "%03d" "${SEG}")
MASSMAP_FILE="${PIN_DIR}/pinocchio.${RUN_MODEL}.massmap.seg${SEG3}.fits"

# Output tag from filename
TAG="$(echo "${NPZ_NAME}" | sed -E 's/^plc_shell_thetaphizmass_//; s/\.npz$//')"
OUTBASE="outputs_${MODE}"
OUTDIR_REL="${OUTBASE}/${TAG}"          # RELATIVE on purpose (python joins with --base-outdir)
OUTDIR_FULL="${OUT_DIR}/${OUTDIR_REL}"

mkdir -p "${OUTDIR_FULL}"

echo -e "\033[32m[Job]\033[0m MODE					  = ${MODE}"
echo -e "\033[32m[Job]\033[0m DISABLE_BARY			  = ${DISABLE_BARY}"
echo -e "\033[32m[Job]\033[0m HALO_NSIDE			  = ${HALO_NSIDE:-<base>}"
echo -e "\033[32m[Job]\033[0m TASK_ID				  = ${TASK_ID}"
echo -e "\033[32m[Job]\033[0m NPZ					  = ${NPZ_NAME}"
echo -e "\033[32m[Job]\033[0m SEG					  = ${SEG3}"
echo -e "\033[32m[Job]\033[0m MAP					  = $(basename "${MASSMAP_FILE}")"
echo -e "\033[32m[Job]\033[0m OUTDIR				  = ${OUTDIR_FULL}"
echo -e "\033[32m[Job]\033[0m PYTHON				  = $(which python)"
echo -e "\033[32m[Job]\033[0m PYTHON VERSION		  = $(python -V)"

# ----------------------------
# User parameters (edit once here)
# ----------------------------
PROFILE_MASS_CUT="1e13"     # particle-mode profiles ONLY
M_BARY_MIN="1e13"           # map-mode baryonification threshold
MIS_NMULT="1.0"             # particle-mode miscentering threshold multiplier

PARTICLE_RMAX_COM="30.0"    # cMpc/h
PAINT_RVIR_FACTOR="1.0"     # map-mode: up to 3*R200m (code default is 3.0 if not overridden)

DEPTH_CYL="50.0"            # cMpc/h
PROFILE_RMAX="10.0"         # cMpc/h
PROFILE_RMIN="0.01"         # cMpc/h
NBINS="20"

# Performance knobs
CHUNK_SIZE_PARTICLE="2048"
CHUNK_SIZE_MAP="64"
POOL_CHUNKSIZE="8"

PROCS="${SLURM_CPUS_PER_TASK}"
PROFILE_PROCS="${SLURM_CPUS_PER_TASK}"

# Choose chunk size by mode
if [[ "${MODE}" == "map" ]]; then
  CHUNK_SIZE="${CHUNK_SIZE_MAP}"
else
  CHUNK_SIZE="${CHUNK_SIZE_PARTICLE}"
fi

# ----------------------------
# Build args
# ----------------------------
EXTRA_ARGS=(--base-outdir "${PIN_DIR}" --outdir "${OUTDIR_FULL}")

# Map mode adds massmap (and optional halo NSIDE)
if [[ "${MODE}" == "map" ]]; then
  EXTRA_ARGS+=(--massmap-file "${MASSMAP_FILE}")
  if [[ -n "${HALO_NSIDE}" ]]; then
    EXTRA_ARGS+=(--halo-nside "${HALO_NSIDE}")
  fi
fi

# Optional baryonification toggle (map mode only)
if [[ "${DISABLE_BARY}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-baryonification)
fi

# ----------------------------
# Run
# ----------------------------
srun python -u "${PY_SCRIPT}" \
  --npz-file "${NPZ_FILE}" \
  "${EXTRA_ARGS[@]}" \
  --paramfile-path "${PARAM_PATH}" \
  --m-bary-min "${M_BARY_MIN}" \
  --profile-mass-cut "${PROFILE_MASS_CUT}" \
  --miscenter-nmult "${MIS_NMULT}" \
  --particle-rmax-com "${PARTICLE_RMAX_COM}" \
  --paint-rvir-factor "${PAINT_RVIR_FACTOR}" \
  --depth-cyl "${DEPTH_CYL}" \
  --profile-rmax "${PROFILE_RMAX}" \
  --profile-rmin "${PROFILE_RMIN}" \
  --nbins "${NBINS}" \
  --chunk-size "${CHUNK_SIZE}" \
  --pool-chunksize "${POOL_CHUNKSIZE}" \
  --procs "${PROCS}" \
  --profile-procs "${PROFILE_PROCS}" \
  --progress-to-stdout

echo -e "\033[32m[Job]\033[0m finished"
