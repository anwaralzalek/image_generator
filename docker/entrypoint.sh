#!/usr/bin/env bash
#
# Dispatch container arguments to the CLI, while still allowing arbitrary
# commands (pytest, python, sh) for verification and one-off maintenance.
#
#   docker run animegen                          -> animegen ui --host 0.0.0.0
#   docker run animegen generate "a party"       -> animegen generate "a party"
#   docker run animegen --verbose info           -> animegen --verbose info
#   docker run animegen pytest -q                -> pytest -q
#   docker run animegen python scripts/download_models.py --model best
set -euo pipefail

case "${1:-ui}" in
    # An animegen subcommand, or a global flag preceding one
    # (--verbose/--config/--version/--help).
    generate | ui | info | -*)
        exec animegen "$@"
        ;;
    # Anything else is a command in its own right: pytest, python, bash.
    *)
        exec "$@"
        ;;
esac
