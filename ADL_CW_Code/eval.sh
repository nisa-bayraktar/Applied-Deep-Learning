#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 0-00:30
#SBATCH --account comsm0045
#SBATCH --mem 100GB
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"
#command to execute python program
python evaluation.py --preds preds.pkl --gts val.pkl
