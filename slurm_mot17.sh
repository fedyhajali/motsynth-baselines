#!/bin/bash
#SBATCH --job-name=tracktor-mot17
#SBATCH --output=output/tracktor-mot17/out.txt
#SBATCH --error=output/tracktor-mot17/err.txt
#SBATCH --chdir=/homes/matteo/NAS/PycharmProjects/motsynth-baselines
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=prod
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=8G
#SBATCH --exclude=aimagelab-srv-00,aimagelab-srv-10

source activate motsynth

python3 -u tools/test_tracktor.py