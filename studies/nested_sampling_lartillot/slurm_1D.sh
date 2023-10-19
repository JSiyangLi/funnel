#!/bin/bash
#
#SBATCH --job-name=lartillot_1d
#SBATCH --output=logs/lartillot_1d.log
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=1

module load gcc/11.3.0 openmpi/4.1.4 r/4.2.1


# Run the script N times
for i in {1..100}
do
  Rscript ns_sim_study_fi.R 1 $i
done


