#!/bin/bash
#
#SBATCH --job-name=lartillot_20d
#SBATCH --output=logs/lartillot_20d_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --time=00:100:00
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=1
#SBATCH --array=0-5

module load python-scientific/3.10.4-foss-2022a
source /fred/oz980/avajpeyi/envs/milan_venv/bin/activate

python run_lartillot.py -d 20 -r 9

