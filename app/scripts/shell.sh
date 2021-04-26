#! /usr/bin/env bash
# shellcheck disable=SC2139
set -euo pipefail
shopt -s expand_aliases

## SETUP ###############################################################################

COMPOSE_PROJECT_NAME="mpi_ml"

HELP="Run bash or any command inside the container.

Usage:
  $(basename "$0") [options] [COMMANDS...]

Options:
  -h. --help   Show this help text"

## ALIASES #############################################################################

CONTAINER_NAME="${COMPOSE_PROJECT_NAME}_shell"

alias build="docker-compose build -q --force-rm $COMPOSE_PROJECT_NAME"
alias start="docker-compose run -d --no-deps --rm --name $CONTAINER_NAME $COMPOSE_PROJECT_NAME sleep infinity"
alias get_cid="docker ps -a -q -f 'name=$CONTAINER_NAME'"
alias execute='docker exec -ite "TERM=$TERM" $(get_cid)'
alias remove='docker rm -f $(get_cid)'

run_cmd="/bin/bash"

## ARGS ################################################################################

ARGS=

while (( $# )); do
    case "$1" in
        # -h or --help show help
        "-h"|"--help")
            echo "$HELP" >&2
            exit 0
            ;;
        # argument as the name of the linter to run only that linter
        *)
            run_cmd=
            ARGS="$ARGS $1"
    esac
    shift
done

function run() {
    eval "execute $run_cmd $ARGS"
}

## TRAP ################################################################################

remove_if_running() {
    CID=$(get_cid)
    if [[ -n "$CID" ]]; then
        remove >/dev/null
    fi
}
trap remove_if_running 0

## MAIN ################################################################################

remove_if_running
build
start
run
