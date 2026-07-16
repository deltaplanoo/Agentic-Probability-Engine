#!/usr/bin/env bash
# Starts the FastMCP server in the background, waits until it's ready, then
# runs the agent in the foreground.
# Stops the server when the agent exits
#
# Uses bash-only features (process substitution), so re-exec under a real
# bash invocation if run via `sh run.sh` (which ignores the shebang above and,
# on macOS, still runs a bash binary but in POSIX mode — so $BASH_VERSION
# alone isn't a reliable check). The sentinel var prevents re-exec looping.
if [ -z "${__RUN_SH_REEXEC:-}" ]; then
    export __RUN_SH_REEXEC=1
    exec bash --noprofile --norc "$0" "$@"
fi

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

export PYTHONUNBUFFERED=1

BLUE=$'\033[0;34m'
RED=$'\033[0;31m'
RESET=$'\033[0m'

# Prefixes each line from stdin with a colored [LABEL] tag. Implemented in
# Python (rather than sed/awk) so behavior is identical on BSD and GNU
# userlands.
tag() {
    local color="$1" label="$2"
    "$PYTHON" -u -c "
import sys
color, label, reset = sys.argv[1], sys.argv[2], sys.argv[3]
for line in sys.stdin:
    sys.stdout.write(f'{color}[{label}]{reset} {line}')
    sys.stdout.flush()
" "$color" "$label" "$RESET"
}

if [ ! -f .env ] && [ -z "${GOOGLE_API_KEY:-}${GEMINI_API_KEY:-}" ]; then
    echo "Warning: no .env file found and GOOGLE_API_KEY/GEMINI_API_KEY not set." >&2
    echo "Copy .env.example to .env and add your key first." >&2
fi

echo "Starting FastMCP server..."
"$PYTHON" src/snap4agentic_advisor_experimental.py > >(tag "$BLUE" SERVER) 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping FastMCP server (pid $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Waiting for the server to come up on port 8000..."
for _ in $(seq 1 30); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server exited before becoming ready — see [SERVER] output above." >&2
        exit 1
    fi
    if (exec 3<>/dev/tcp/localhost/8000) 2>/dev/null; then
        exec 3<&- 3>&-
        break
    fi
    sleep 0.5
done

echo "Server is up. Running agent..."
"$PYTHON" src/agent.py 2>&1 | tag "$RED" AGENT
