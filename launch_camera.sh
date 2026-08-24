#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
bash "$SCRIPT_DIR/scripts/launch_camera.sh" "$@"
