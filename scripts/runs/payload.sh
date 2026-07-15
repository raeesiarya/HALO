#!/bin/bash

set -euo pipefail

uv sync

srun uv run lmlm-audit --wandb-activation on
