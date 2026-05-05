#!/bin/sh
#SBATCH --job-name=helper_data_comparison
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=node0[21-40],node0[45-56]
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
source /home/sfaour/venv/bin/activate

# Set the working directory
cd /home/sfaour/MyWork/SRAM-PUF-key-generation


python -m nvm_free_tmvs.experiments.helper_data_comparator
deactivate