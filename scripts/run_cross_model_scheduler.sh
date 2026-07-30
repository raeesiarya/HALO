#!/usr/bin/env bash
# Run the cross-model evaluation as a dependency-aware GPU job graph.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec uv run python -m halo.scheduler "$@"
