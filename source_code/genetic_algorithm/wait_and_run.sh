#!/bin/bash
export PATH="/home/juhyeong/miniconda/envs/sem_env/bin:$PATH"

# PID of the currently running 15% GA
TARGET_PID=3616809

echo "Waiting for PID $TARGET_PID to finish..."
while kill -0 $TARGET_PID 2>/dev/null; do
    sleep 60
done
echo "PID $TARGET_PID has finished."

# Replace the convergence threshold
echo "Updating convergence threshold in Genetic_Algorithm.py to 0.005..."
sed -i 's/0.0075/0.005/g' Genetic_Algorithm.py

# Run 10% seed 50
echo "Starting 10% seed 50"
python Genetic_Algorithm.py --seed 50 --electrification_rate 1.0 --trial_num 1 > ga_10_seed50.log 2>&1

# Run 10% seed 51
echo "Starting 10% seed 51"
python Genetic_Algorithm.py --seed 51 --electrification_rate 1.0 --trial_num 1 > ga_10_seed51.log 2>&1

echo "All extra tasks completed."
