#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --mem=1GB

module load Python/3.9.6-GCCcore-11.2.0

source .venv/bin/activate

python bluff-game/test_three_first_order.py