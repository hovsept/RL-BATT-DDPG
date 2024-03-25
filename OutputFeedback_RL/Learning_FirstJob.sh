#!/bin/bash
#SBATCH --job-name="Learning_FirstJob"
#SBATCH --time=01:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=Education-ME-MSc-SC
module load 2023r1
module load openmpi
module load python
module load py-numpy
srun python main_training_testing_output_fdbk.py > training.log