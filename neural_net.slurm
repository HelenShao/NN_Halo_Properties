#!/bin/bash
#SBATCH --job-name=halo_mass  # change name depending on property you are mapping to
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=hshao@princeton.edu

module purge
module load anaconda3
conda activate pytorch-gpu

python main.py
