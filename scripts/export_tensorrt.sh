#!/bin/bash
# =====================================================================
# JetRacer AI — TensorRT Engine Exporter (trtexec)
# Usage:
#   bash scripts/export_tensorrt.sh models/urban_traffic/best.onnx models/urban_traffic/best.engine
# =====================================================================

ONNX_INPUT="${1:-models/urban_traffic/best.onnx}"
ENGINE_OUTPUT="${2:-models/urban_traffic/best.engine}"

echo -e "\e[1;34m[*] Exporting ONNX -> TensorRT FP16 Engine...\e[0m"
echo "Input : $ONNX_INPUT"
echo "Output: $ENGINE_OUTPUT"

# Locate trtexec binary
TRTEXEC_BIN=""
if command -v trtexec &> /dev/null; then
    TRTEXEC_BIN="trtexec"
elif [ -f "/usr/src/tensorrt/bin/trtexec" ]; then
    TRTEXEC_BIN="/usr/src/tensorrt/bin/trtexec"
else
    TRTEXEC_BIN=$(find /usr -name trtexec 2>/dev/null | head -n 1)
fi

if [ -z "$TRTEXEC_BIN" ]; then
    echo -e "\e[1;31m[ERROR] trtexec binary not found! Please check TensorRT installation.\e[0m"
    exit 1
fi

echo -e "\e[1;32m[*] Found trtexec at: $TRTEXEC_BIN\e[0m"
$TRTEXEC_BIN --onnx="$ONNX_INPUT" --saveEngine="$ENGINE_OUTPUT" --fp16

echo -e "\n\e[1;32m[✓] TensorRT Engine Export Completed: $ENGINE_OUTPUT\e[0m"
