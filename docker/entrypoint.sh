#!/bin/sh
# With no arguments: fetch SECOM if it is missing, then run the full pipeline
# (PNG figures). With arguments: run that `secom` subcommand.
set -e

if [ "$#" -eq 0 ]; then
    secom fetch
    exec secom run
fi

exec secom "$@"
