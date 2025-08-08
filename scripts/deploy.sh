#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-sagarpawar123/housing-api:latest}"

echo "🚀 Pulling image: $IMAGE"
docker pull "$IMAGE"

echo "🛑 Removing old container (if any)"
docker rm -f housing-api >/dev/null 2>&1 || true

echo "🚢 Running container on :8000"
docker run -d --name housing-api -p 8000:8000 "$IMAGE"

echo "✅ Deployed → http://localhost:8000"

