#!/bin/bash
#SBATCH --job-name=FI_array
#SBATCH --output=logs/FI_%A_%a.out
#SBATCH --error=logs/FI_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=0:30:00
#SBATCH --array=1-5

module load gcc/13.2.0


block_values=(1 3 7 15 30)
block_index=$(((SLURM_ARRAY_TASK_ID - 1) % 5 + 1))
# Get the values of ppb and block for this task
block=${block_values[$block_index - 1]}

# Run your C++ program with the specified ppb and block values
./fi_blocked.o --PPB 3 --BLOCK $block --V 1 --R 100 --N 10000000
