#! /usr/bin/env bash
# shellcheck disable=SC2139
set -euo pipefail
shopt -s expand_aliases

## SETUP ###############################################################################

APPDIR=$(dirname "$0")/..
ROOT="$APPDIR/src"
CONFIG_FILE="$APPDIR/../setup.cfg"
COMPOSE_PROJECT_NAME="mpi_ml"

HELP="Run linters.

Usage:
  $(basename "$0") [options] [LINTERS...]

Options:
  -h. --help    Show this help text
  -l, --local   Run linters locally instead of in the '$COMPOSE_PROJECT_NAME' service's container
  -c, --check   Run linters in check mode, which will NOT try to automatically fix the code if any error is found"

## PRINTS ##############################################################################

YELLOW='\033[1;33m'
NC='\033[0m'

function print_header() {
    MSG="$1"
    echo -e "\n${YELLOW}* ${MSG}${NC}"
}

## ALIASES #############################################################################

CONTAINER_NAME="${COMPOSE_PROJECT_NAME}_linters"

alias build="docker-compose build -q --force-rm $COMPOSE_PROJECT_NAME"
alias start="docker-compose run -d --no-deps --rm --name $CONTAINER_NAME $COMPOSE_PROJECT_NAME sleep infinity"
alias get_cid="docker ps -a -q -f 'name=$CONTAINER_NAME'"
alias execute='docker exec -ite "TERM=$TERM" $(get_cid)'
alias remove='docker rm -f $(get_cid)'

function define_local_aliases() {
    alias build=
    alias start=
    alias get_cid=
    alias execute=
    alias remove=
    echo "Running locally..."
}

function define_check_linters(){
    alias run_linter_black="print_header BLACK && execute black --check $ROOT"
    alias run_linter_isort="print_header ISORT && execute isort --settings $CONFIG_FILE --check $ROOT && echo All done! ✨ 🍪 ✨"
    alias run_linter_autoflake=""
}

alias run_linter_black="print_header BLACK && execute black $ROOT"
alias run_linter_isort="print_header ISORT && execute isort --settings $CONFIG_FILE $ROOT && echo All done! ✨ 🍪 ✨"
alias run_linter_autoflake="print_header AUTOFLAKE && execute autoflake -r -i --remove-all-unused-imports --remove-unused-variables $ROOT && echo All done! ✨ 🍬 ✨"
alias run_linter_flake8="print_header FLAKE8 && execute flake8 --config $CONFIG_FILE $ROOT && echo All done! ✨ 🍩 ✨"
alias run_linter_bandit="print_header BANDIT && execute bandit -r $ROOT && echo All done! ✨ 🧁 ✨"

## ARGS ################################################################################

read -r -a run_all_linters <<< "$(compgen -a | grep '^run_' | xargs)"
declare -a run_spec_linters

while (( $# )); do
    case "$1" in
        # -h or --help show help
        "-h"|"--help")
            echo "$HELP" >&2
            exit 0
            ;;
        # -l or --local argument to run on your local machine
        "-l"|"--local")
            define_local_aliases
            ;;
        # -c or --check argument to run in check mode
        "-c"|"--check")
            define_check_linters
            ;;
        # argument as the name of the linter to run only that linter
        *)
            run_all_linters=()
            run_spec_linters+=("run_linter_$1")
    esac
    shift
done

run_linter_cmds=( "${run_all_linters[@]+${run_all_linters[@]}}" "${run_spec_linters[@]+${run_spec_linters[@]}}" )

function run_linters() {
    RC=0
    for run_linter_cmd in "${run_linter_cmds[@]}"; do
        eval "$run_linter_cmd" || RC=$?
    done
    exit ${RC}
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
run_linters
