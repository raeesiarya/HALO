#!/usr/bin/env bash
# Run the 35-job cross-model evaluation with dependency-aware GPU scheduling.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec uv run python -m halo.scheduler "$@"
