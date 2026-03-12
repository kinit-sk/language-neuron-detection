#!/bin/bash
#SBATCH --account=p904-24-3
#SBATCH --mail-user=<jakub.kopal@kinit.sk>
#SBATCH --time=09:00:00 # Estimate to increase job priority

## Nodes allocation
#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate language-neuron-detection 

EXPERIMENT_CONFIG="7_finetuning_latn.yaml"
MLFLOW_PORT="${MLFLOW_PORT:-5002}"
MLFLOW_BACKEND_URI="${MLFLOW_BACKEND_URI:-sqlite:///mlruns/mlruns.db}"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-file:mlruns}"
MLFLOW_TRACKING_URI="http://127.0.0.1:${MLFLOW_PORT}"
JOB_LOG_DIR="${JOB_LOG_DIR:-logs/7_finetuning_latn}"
mkdir -p "${JOB_LOG_DIR}"
MLFLOW_LOG_PATH="${MLFLOW_LOG_PATH:-${JOB_LOG_DIR}/mlflow-${SLURM_JOB_ID:-$$}.log}"
MLFLOW_STARTUP_TIMEOUT="${MLFLOW_STARTUP_TIMEOUT:-60}"

cleanup() {
  echo "Stopping MLflow server..."
  if [[ -n "${MLFLOW_TAIL_PID:-}" ]]; then
    kill "$MLFLOW_TAIL_PID" 2>/dev/null || true
  fi
  if [[ -n "${MLFLOW_PID:-}" ]]; then
    kill "$MLFLOW_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

wait_for_mlflow() {
  local start_ts
  start_ts=$(date +%s)

  echo "Waiting for MLflow server to be ready..."
  while true; do
    if ! kill -0 "$MLFLOW_PID" 2>/dev/null; then
      echo "MLflow server exited before becoming ready."
      echo "Last MLflow log lines from ${MLFLOW_LOG_PATH}:"
      tail -n 50 "${MLFLOW_LOG_PATH}" || true
      return 1
    fi

    if python - "$MLFLOW_PORT" <<'PY'
import sys
from urllib.error import URLError
from urllib.request import urlopen

port = sys.argv[1]
for path in ("/health", "/"):
    try:
        with urlopen(f"http://127.0.0.1:{port}{path}", timeout=1) as response:
            if response.status < 500:
                raise SystemExit(0)
    except URLError:
        pass
    except Exception:
        pass
raise SystemExit(1)
PY
    then
      return 0
    fi

    if (( $(date +%s) - start_ts >= MLFLOW_STARTUP_TIMEOUT )); then
      echo "Timed out waiting ${MLFLOW_STARTUP_TIMEOUT}s for MLflow server."
      echo "Last MLflow log lines from ${MLFLOW_LOG_PATH}:"
      tail -n 50 "${MLFLOW_LOG_PATH}" || true
      return 1
    fi

    sleep 1
  done
}

echo "Starting MLflow server..."
mlflow server \
  --backend-store-uri "$MLFLOW_BACKEND_URI" \
  --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port "$MLFLOW_PORT" \
  >"${MLFLOW_LOG_PATH}" 2>&1 &

MLFLOW_PID=$!
tail -n +1 -f "${MLFLOW_LOG_PATH}" &
MLFLOW_TAIL_PID=$!

wait_for_mlflow

echo "MLflow server running at ${MLFLOW_TRACKING_URI} on $(hostname) (PID=${MLFLOW_PID}) logs in ${MLFLOW_LOG_PATH}"
echo "Launching runexp config: ${EXPERIMENT_CONFIG}"


python -m runexp -cn "${EXPERIMENT_CONFIG}" "run_args.tracker.tracking_uri=${MLFLOW_TRACKING_URI}"
