#!/bin/bash
#SBATCH --job-name=safevla-monitor
#SBATCH --output=/network/scratch/d/deschaer/SafeVLA/slurm_logs/monitor-%j.out
#SBATCH --error=/network/scratch/d/deschaer/SafeVLA/slurm_logs/monitor-%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=unkillable-cpu

cd /network/scratch/d/deschaer/SafeVLA
bash scripts/monitor.sh
