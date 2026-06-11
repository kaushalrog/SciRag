#!/bin/bash
set -e

echo "=== SciRAG-UQ Experiment Pipeline ==="

echo "1. Generating Datasets..."
python3 src/data/dataset_builder.py

echo "2. Running Core Baselines and SciRAG-UQ..."
python3 src/evaluation/experiment_runner.py

echo "3. Running Threshold Sweep (Abstention analysis)..."
python3 src/evaluation/threshold_sweep.py

echo "All experiments completed successfully. Check the results/ directory for outputs and plots."
