#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python -m origami_prediction_visualizer "$@"
