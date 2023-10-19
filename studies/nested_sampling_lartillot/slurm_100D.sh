#!/bin/bash
#
#SBATCH --job-name=lartillot_100d
#SBATCH --output=logs/lartillot_100d_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=1
#SBATCH --array=0-100

module load gcc/11.3.0 openmpi/4.1.4 r/4.2.1

Rscript ns_sim_study_fi.R -d 100 -s $SLURM_ARRAY_TASK_ID