# Sourced by the setup scripts (setup_colmlm.sh, setup_nulls.sh,
# setup_data.sh). `detach_setup NAME "$@"` re-runs the calling script in the
# background under nohup — it survives the SSH session closing and prints
# nothing to the terminal — with all output in logs/NAME.log, then exits the
# foreground copy after printing how to follow and stop it.
#
#   HALO_SETUP_FOREGROUND=1   run attached instead
#
# The detached copy runs with HALO_SETUP_DETACHED=1 and brackets its log with
# start/finish lines; the finish line carries the exit status. Children it
# starts inherit the flag (setup_data.sh relies on this for its halves' logs).

detach_setup() {
    local name="$1"
    shift
    if [ "${HALO_SETUP_DETACHED:-0}" = "1" ]; then
        echo "=== $name started $(date '+%F %T') (PID $$) ==="
        trap 'status=$?; echo; echo "=== '"$name"' finished $(date "+%F %T") (exit $status) ==="' EXIT
        return 0
    fi
    [ "${HALO_SETUP_FOREGROUND:-0}" = "1" ] && return 0

    local log_dir="$REPO_ROOT/logs"
    local log="$log_dir/$name.log"
    local pid_file="$log_dir/$name.pid"
    mkdir -p "$log_dir"
    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "error: $name is already running (PID $(cat "$pid_file"))." >&2
        echo "       follow it: tail -F $log" >&2
        exit 1
    fi
    : > "$log"
    # setsid (Linux) puts the run in its own process group, so one kill also
    # stops the uv/python children; macOS has no setsid, plain nohup there.
    local launcher=(nohup)
    command -v setsid >/dev/null 2>&1 && launcher=(setsid nohup)
    HALO_SETUP_DETACHED=1 "${launcher[@]}" bash "$0" "$@" </dev/null >"$log" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"
    disown 2>/dev/null || true
    echo "Started $name in the background (PID $pid); nothing prints here."
    echo "  follow:  tail -F $log"
    if [ "${launcher[0]}" = "setsid" ]; then
        echo "  stop:    kill -- -$pid"
    else
        echo "  stop:    kill $pid"
    fi
    exit 0
}
