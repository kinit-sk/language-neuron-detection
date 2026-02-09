#!/usr/bin/env bash
set -euo pipefail
MLFLOW_PORT=5001
MLFLOW_BACKEND_URI="sqlite:///mlruns/mlruns.db"
MLFLOW_ARTIFACT_ROOT="file:mlruns"

cleanup() {
  echo "Stopping MLflow server..."
  if [[ -n "${MLFLOW_PID:-}" ]]; then
    kill "$MLFLOW_PID" 2>/dev/null || true
  fi
}
# Ensure cleanup happens on script exit, error, or Ctrl+C
trap cleanup EXIT INT TERM

# # Start mlflow server
# mlflow server --backend-store-uri=sqlite:///mlruns/mlruns.db --default-artifact-root=file:mlruns --host 0.0.0.0 --port 5001 >/tmp/mlflow.log 2>&1 &

echo "Starting MLflow server..."
mlflow server \
  --backend-store-uri "$MLFLOW_BACKEND_URI" \
  --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port "$MLFLOW_PORT" \
  >/tmp/mlflow.log 2>&1 &

MLFLOW_PID=$!

# Optional but recommended: wait until server is actually up
echo "Waiting for MLflow server to be ready..."
until curl -s "http://127.0.0.1:$MLFLOW_PORT" >/dev/null; do
  sleep 0.5
done

echo "MLflow server running (PID=$MLFLOW_PID)"



# Finetuning entrypoint
# See: 
#   1) configs/example.yaml for finetuning configs
#   2) finetuing_impl/example.yaml for finetuing implementation
python -m runexp -cn example.yaml
