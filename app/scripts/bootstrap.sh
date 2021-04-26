#! /usr/bin/env bash
set -euo pipefail

YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}* Create virtual env (you should have Python 3.8 installed)${NC}"
pip install virtualenv
virtualenv --python=python.38 .venv && \
    . .venv/bin/activate && \
    pip install -r requirements-dev.txt

echo -e "${YELLOW}* Installing requirements-dev.txt${NC}"
pip install -r requirements-dev.txt

echo -e "${YELLOW}* Installing pre-commit${NC}"
pre-commit install --config .pre-commit-config.yaml

echo -e "${YELLOW}* Setting up .env${NC}"
echo -e "${YELLOW}* WARNING: You still need to add the VARIABLES${NC}"
cp .env.default .env
