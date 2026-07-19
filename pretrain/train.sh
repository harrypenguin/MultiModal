#!/bin/bash
#SBATCH --nodes=4
#SBATCH --time=10:20:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=desi_g
#SBATCH --gpus-per-node=4

conda activate /global/cfs/cdirs/desi/users/pzehao/envs/peng
srun --ntasks-per-node=4 --cpus-per-task=32 python MaeTrain.py