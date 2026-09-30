#!/bin/bash
export PATH="/home/juhyeong/miniconda/envs/sem_env/bin:$PATH"

echo "Starting 15% seed 47"
python Genetic_Algorithm.py --seed 47 --electrification_rate 1.5 --trial_num 6
echo "Finished 15% seed 47"

echo "Starting 20% seed 47"
python Genetic_Algorithm.py --seed 47 --electrification_rate 2.0 --trial_num 6
echo "Finished 20% seed 47"

echo "Starting 2% seed 47"
python Genetic_Algorithm.py --seed 47 --electrification_rate 0.2 --trial_num 6
echo "Finished 2% seed 47"

echo "Starting 5% seed 47"
python Genetic_Algorithm.py --seed 47 --electrification_rate 0.5 --trial_num 6
echo "Finished 5% seed 47"

echo "All tasks for seed 47 completed."
