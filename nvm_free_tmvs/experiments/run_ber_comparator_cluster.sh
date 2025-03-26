#!/bin/sh
#SBATCH --job-name=ber_comparison
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
source /home/sfaour/venv/bin/activate

# Set the working directory
cd /home/sfaour/MyWork/TMVS-for-Robust-SRAM-PUFs


python -m nvm_free_tmvs.experiments.global_ber_processor
deactivate