#!/bin/sh
#SBATCH --job-name=grid_search_odhd
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
source /home/sfaour/venv/bin/activate

# Set the working directory
cd /home/sfaour/MyWork/SRAM-PUF-key-generation


python -m previous_work.helperless_stabilizer_bernardini.experiments.global_ber_processor
deactivate