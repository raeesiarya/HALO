#!/bin/bash
set -euo pipefail

##############################################
# Require and load .env
##############################################
ENV_FILE="${ENV_FILE:-.env}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found"
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

##############################################
# Validate required variables
##############################################
: "${ACCOUNT:?Missing ACCOUNT in .env}"
: "${WALL_MIN:?Missing WALL_MIN in .env}"
: "${WALL:?Missing WALL in .env}"
: "${LOG_DIR:?Missing LOG_DIR in .env}"

: "${CPUS_A5000:?Missing CPUS_A5000 in .env}"
: "${CPUS_L40:?Missing CPUS_L40 in .env}"
: "${CPUS_A40:?Missing CPUS_A40 in .env}"
: "${CPUS_2080TI:?Missing CPUS_2080TI in .env}"
: "${CPUS_1080TI:?Missing CPUS_1080TI in .env}"

mkdir -p "$LOG_DIR"

declare -a JOBS=()

##############################################
# submit(...)
##############################################
submit() {
    local partition="$1"
    local qos="$2"
    local gres="$3"
    local gpus="$4"
    local cpt="$5"

    sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$partition" \
        --qos="$qos" \
        --time-min="$WALL_MIN" \
        --time="$WALL" \
        --nodes=1 \
        --gres="$gres" \
        --ntasks-per-node="$gpus" \
        --cpus-per-task="$cpt" \
        --output="${LOG_DIR}/slurm-%j.out" \
        --error="${LOG_DIR}/slurm-%j.err" \
        scripts/runs/payload.sh
}

##############################################
# Submit jobs
##############################################
echo "Submitting multi-partition GPU candidates..."

jid=$(submit "savio4_gpu" "a5k_gpu4_normal" "gpu:A5000:1" "1" "$CPUS_A5000")
echo "A5000 -> $jid"
JOBS+=("$jid")

jid=$(submit "savio4_gpu" "savio_lowprio" "gpu:L40:1" "1" "$CPUS_L40")
echo "L40 -> $jid"
JOBS+=("$jid")

jid=$(submit "savio3_gpu" "a40_gpu3_normal" "gpu:A40:1" "1" "$CPUS_A40")
echo "A40 -> $jid"
JOBS+=("$jid")

jid=$(submit "savio3_gpu" "gtx2080_gpu3_normal" "gpu:GTX2080TI:1" "1" "$CPUS_2080TI")
echo "2080Ti -> $jid"
JOBS+=("$jid")

jid=$(submit "savio2_1080ti" "savio_normal" "gpu:GTX1080TI:1" "1" "$CPUS_1080TI")
echo "1080Ti -> $jid"
JOBS+=("$jid")

##############################################
# Wait for first RUNNING job
##############################################
echo
echo "Waiting for first job to start..."

WINNER=""

while [[ -z "$WINNER" ]]; do
    sleep 5

    running_jobs=()
    for jid in "${JOBS[@]}"; do
        state=$(squeue -j "$jid" -h -o "%T" 2>/dev/null || true)
        if [[ "$state" == "RUNNING" ]]; then
            running_jobs+=("$jid")
        fi
    done

    if (( ${#running_jobs[@]} > 0 )); then
        WINNER="${running_jobs[0]}"
    fi
done

##############################################
# Cancel others
##############################################
echo "Winner: $WINNER"
echo "Cancelling others..."

for jid in "${JOBS[@]}"; do
    if [[ "$jid" != "$WINNER" ]]; then
        scancel "$jid" 2>/dev/null || true
    fi
done

echo "Done."