#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH --partition=nvgpu
#SBATCH --account=m05
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -J afno
#SBATCH -o slurm-%x.%j.out

environment=$(realpath env/ngc-pytorch-22.04.toml)

config_file=./config/AFNO.yaml
config='afno_backbone'
run_num=$SLURM_JOBID

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=INFO

export MASTER_ADDR=$(hostname)

set -x
srun -u --environment=$environment bash -c "
    pip install h5py mpi4py wandb ruamel.yaml tqdm timm einops
    source export_DDP_vars.sh
    python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
"
