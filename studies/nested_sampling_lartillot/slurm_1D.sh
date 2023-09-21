#!/bin/bash
#
#SBATCH --job-name=lartillot_1d_%A_%a
#SBATCH --output=logs/lartillot_1d_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=1
#SBATCH --array=0-5

module load python-scientific/3.10.4-foss-2022a
source /fred/oz980/avajpeyi/envs/milan_venv/bin/activate

ARRAY_ARGS=(0 1 2 3 4)

python run_lartillot.py -d 1

