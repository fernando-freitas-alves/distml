#!/usr/bin/env bash
set -u

APPDIR=$(dirname "$0")/..

"$APPDIR/src/main.py" --helpfull
