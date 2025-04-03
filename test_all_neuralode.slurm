#!/bin/sh
#SBATCH -J benchmark_neuralODE
#SBATCH -t 06:00:00
#SBATCH -p gpu_h100
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
cd /home/gijsv/neuralode_benchmarking
COMMAND="/home/gijsv/.conda/envs/jax/bin/python /home/gijsv/test_neuralode/neuralode_benchmarking.py"
# Baselines on GPU:
$COMMAND -d "gpu" -n 1 -m "shard" -c "pid"
$COMMAND -d "gpu" -n 1 -m "shard" -c "stepto"
# Benchmark multi-GPU:
$COMMAND -d "gpu" -n 2 -m "pmap" -c "pid"
$COMMAND -d "gpu" -n 2 -m "shard" -c "pid"
$COMMAND -d "gpu" -n 2 -m "shmap" -c "pid"
$COMMAND -d "gpu" -n 2 -m "pmap" -c "stepto"
$COMMAND -d "gpu" -n 2 -m "shard" -c "stepto"
$COMMAND -d "gpu" -n 2 -m "shard" -c "stepto"
# Baselines for CPU:
$COMMAND -d "cpu" -n 1 -m "shard" -c "pid"
$COMMAND -d "cpu" -n 1 -m "shard" -c "stepto"
# Benchmark multiprocessing:
$COMMAND -d "cpu" -n 8 -m "pmap" -c "pid"
$COMMAND -d "cpu" -n 8 -m "shard" -c "pid"
$COMMAND -d "cpu" -n 8 -m "shmap" -c "pid"
$COMMAND -d "cpu" -n 8 -m "pmap" -c "stepto"
$COMMAND -d "cpu" -n 8 -m "shard" -c "stepto"
$COMMAND -d "cpu" -n 8 -m "shmap" -c "stepto"
