#!/bin/bash
#SBATCH --job-name=morse
#SBATCH --output=out.txt
#SBATCH --account=lr_ninjaone
#SBATCH --partition=es1
#SBATCH --qos=condo_ninjaone_es1
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --constraint es1_2080ti ##(other options: es1_v100, es1_1080ti, es1_2080ti)
source activate newtonnet
python ~/NewtonNet/cli/newtonnet_train -c ~/20230120_AnalPES/config.yml
