#!/usr/bin/env bash
set -euo pipefail

# --- Configurable via env ---
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_SUBDIR="${LOG_SUBDIR:-logs}"
# Optional manual override, e.g. TRAIN_PY_REL="multiloss_pdbbind/final_test.py"
TRAIN_PY_REL="${TRAIN_PY_REL:-}"

# --- Resolve paths relative to this script ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Candidate locations (in order): ../foo.py, ./foo.py, override
CANDIDATES=()
if [[ -z "${TRAIN_PY_REL}" ]]; then
  CANDIDATES+=("${SCRIPT_DIR}/../multi-objective-pdbbind.py")
  CANDIDATES+=("${SCRIPT_DIR}/multi-objective-pdbbind.py")
else
  # Respect manual override (relative to repo/script dir if not absolute)
  if [[ "${TRAIN_PY_REL}" = /* ]]; then
    CANDIDATES+=("${TRAIN_PY_REL}")
  else
    CANDIDATES+=("${SCRIPT_DIR}/${TRAIN_PY_REL}")
    CANDIDATES+=("${PARENT_DIR}/${TRAIN_PY_REL}")
  fi
fi

TRAIN_PY=""
for c in "${CANDIDATES[@]}"; do
  if [[ -f "${c}" ]]; then TRAIN_PY="${c}"; break; fi
done

if [[ -z "${TRAIN_PY}" ]]; then
  echo "✗ Could not find multi-objective-pdbbind.py."
  echo "  Tried:"
  printf '  - %s\n' "${CANDIDATES[@]}"
  echo "Tip: set TRAIN_PY_REL to the correct relative path."
  exit 1
fi

# Logging
LOG_DIR="${PARENT_DIR}/${LOG_SUBDIR}"
mkdir -p "${LOG_DIR}"
TS="$(date +"%Y%m%d_%H%M%S")"
OUT_LOG="${LOG_DIR}/gpu_training_${TS}.out"
ERR_LOG="${LOG_DIR}/gpu_training_${TS}.err"

# Env
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PARENT_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"

# Header
echo "============================================="
echo "Starting job @ $(date)"
echo "Script dir: ${SCRIPT_DIR}"
echo "Parent dir: ${PARENT_DIR}"
echo "Python file: ${TRAIN_PY}"
echo "Logs:"
echo "  OUT -> ${OUT_LOG}"
echo "  ERR -> ${ERR_LOG}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Python: $(${PYTHON_BIN} --version 2>&1 || echo 'python not found')"
if command -v nvidia-smi &>/dev/null; then nvidia-smi || true; fi
if command -v free &>/dev/null; then echo "Initial memory:"; free -h; fi
echo "============================================="

# Run
set +e
${PYTHON_BIN} -u "${TRAIN_PY}" 1> "${OUT_LOG}" 2> "${ERR_LOG}"
EC=$?
set -e

if [[ ${EC} -eq 0 ]]; then
  echo "✓ Training completed successfully."
else
  echo "✗ Training failed (exit ${EC}). See ${ERR_LOG}"
fi
echo "OUT: ${OUT_LOG}"
echo "ERR: ${ERR_LOG}"
if command -v free &>/dev/null; then echo "Final memory:"; free -h; fi
echo "Done @ $(date)"
