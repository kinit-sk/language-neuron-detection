#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_CONFIG="7_finetuning_latn.yaml"
MLFLOW_PORT="${MLFLOW_PORT:-5002}"
MLFLOW_BACKEND_URI="${MLFLOW_BACKEND_URI:-sqlite:///mlruns/mlruns.db}"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-file:mlruns}"
MLFLOW_TRACKING_URI="http://127.0.0.1:${MLFLOW_PORT}"

cleanup() {
  echo "Stopping MLflow server..."
  if [[ -n "${MLFLOW_PID:-}" ]]; then
    kill "$MLFLOW_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "Starting MLflow server..."
mlflow server \
  --backend-store-uri "$MLFLOW_BACKEND_URI" \
  --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port "$MLFLOW_PORT" \
  >/tmp/mlflow.log 2>&1 &

MLFLOW_PID=$!

echo "Waiting for MLflow server to be ready..."
until curl -s "http://127.0.0.1:${MLFLOW_PORT}" >/dev/null; do
  sleep 0.5
done

echo "MLflow server running at ${MLFLOW_TRACKING_URI} (PID=${MLFLOW_PID}) logs in /tmp/mlflow.log"
echo "Launching runexp config: ${EXPERIMENT_CONFIG}"

python -m runexp -cn "${EXPERIMENT_CONFIG}" "run_args.tracker.tracking_uri=${MLFLOW_TRACKING_URI}"
