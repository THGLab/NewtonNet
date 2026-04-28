#!/bin/bash
#SBATCH -A m2834_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -t 04:00:00
#SBATCH -J newtonnet_electrolyte
#SBATCH -o logs/newtonnet_%j.out
#SBATCH -e logs/newtonnet_%j.err

mkdir -p logs

cd $SCRATCH/code/les/NewtonNet/scripts

$SCRATCH/code/les/nnpackages/newtonnet/bin/python newtonnet_train.py --config config_electrolyte.yml
