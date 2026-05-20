#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH -J BALUJA_STEG
#SBATCH -o slurm-%j.out

#SBATCH -p gpu
#SBATCH --gres="gpu:a100:1"
python3 updatedLenslessScript.py