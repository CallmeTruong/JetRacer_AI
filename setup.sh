#!/bin/bash
# Mirror launcher for setup_env.sh & setup_car.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
bash "$SCRIPT_DIR/scripts/setup_env.sh"
bash "$SCRIPT_DIR/scripts/setup_car.sh"