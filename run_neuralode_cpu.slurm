#!/bin/sh
#SBATCH -J benchmark_neuralODE
#SBATCH -t 06:00:00
#SBATCH -p rome
#SBATCH --mem=26G
#SBATCH --cpus-per-task 16
cd /home/gijsv/neuralode_benchmarking
COMMAND="/home/gijsv/.conda/envs/jax/bin/python /home/gijsv/neuralode_benchmarking/neuralode_benchmarking.py"
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
