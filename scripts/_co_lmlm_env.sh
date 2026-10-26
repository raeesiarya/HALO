# Sourced by the Co-LMLM run scripts (after REPO_ROOT is set). Chooses the
# uv project that provides the Co-LMLM runtime:
#   - x86_64 / macOS: the Co-LMLM checkout's own environment (cwd);
#   - Linux/aarch64:  HALO's port at envs/co-lmlm-aarch64 (lil-lab's lock
#     pins x86-only wheels; see that pyproject for why the port is sound).
# CO_LMLM_UV_PROJECT=<dir> forces a project; CO_LMLM_UV_PROJECT= (set but
# empty) forces the checkout's own environment.

if [ -z "${CO_LMLM_UV_PROJECT+x}" ]; then
    if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "aarch64" ]; then
        CO_LMLM_UV_PROJECT="$REPO_ROOT/envs/co-lmlm-aarch64"
    else
        CO_LMLM_UV_PROJECT=""
    fi
fi
export CO_LMLM_UV_PROJECT

# `co_lmlm_uv sync` / `co_lmlm_uv run python ...` in place of `uv ...`.
# --project does not change the working directory, so the checkout's
# relative `src` on PYTHONPATH keeps resolving.
co_lmlm_uv() {
    local subcommand="$1"
    shift
    if [ -n "$CO_LMLM_UV_PROJECT" ]; then
        uv "$subcommand" --project "$CO_LMLM_UV_PROJECT" "$@"
    else
        uv "$subcommand" "$@"
    fi
}
