#!/bin/bash

#SBATCH --job-name=tagger_check-%j
#SBATCH --account=gpu_gres
#SBATCH --output=/work/ext-siddharta/logs/checklog_%j.log
#SBATCH --error=/work/ext-siddharta/logs/checkerror_%j.log
#SBATCH --mem=18G
#SBATCH --cpus-per-task=4
##SBATCH --ntasks=5
#SBATCH --partition=gpu,qgpu
#SBATCH --time=11:59:59
##SBATCH --nodes=4
#SBATCH --gres=gpu:1



source ~/.bashrc
conda deactivate
conda deactivate
conda activate /work/rseidita/miniforge3/envs/gnn_gpu

python basic_example.py