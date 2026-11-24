#!/usr/bin/env bash
# Copy the trained model from your Mac (./models) into the Kubernetes PVC.
# Run from the project root after training (Airflow):
#     ./k8s/load_model.sh
set -euo pipefail

cd "$(dirname "$0")/.."
for f in model.pkl metrics.json predictions.json; do
  [[ -f "models/$f" ]] || { echo "models/$f not found - run the pipeline first"; exit 1; }
done

echo "Waiting for the model-loader pod..."
kubectl wait --for=condition=Ready pod/model-loader --timeout=120s

for f in model.pkl metrics.json predictions.json; do
  # copy to a temp name, then rename -> Flask never reads a half-copied file
  kubectl cp "models/$f" "model-loader:/models/$f.tmp"
  kubectl exec model-loader -- sh -c "chmod 644 /models/$f.tmp && mv /models/$f.tmp /models/$f"
  echo "copied $f"
done

echo
kubectl exec model-loader -- ls -l /models
echo
echo "Model version now in the PVC:"
grep -o '"version": *"[^"]*"' models/metrics.json
echo "Flask pods pick it up automatically on their next request."
