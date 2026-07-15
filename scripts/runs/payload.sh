#!/bin/bash

set -euo pipefail

# Sync dependencies from pyproject.toml
uv sync

# Activate uv-managed venv
source .venv/bin/activate

# Run workload
srun python3 src/lmlm-audit/run_audit.py --wandb_activation on