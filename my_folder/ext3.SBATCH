#!/bin/bash
#SBATCH --output=/scratch/wz3008/SlotAttn/slurm_outputs/%x_%j.out
#SBATCH --error=/scratch/wz3008/SlotAttn/slurm_outputs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wz3008@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=process_ext3
#SBATCH --account torch_pr_36_mren


# cd /scratch/wz3008/;gunzip overlay-50G-10M.ext3.gz;

singularity exec --nv --fakeroot \
    --overlay /scratch/wz3008/overlay-50G-10M-oclf.ext3:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
    -c "
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh; \
    bash /ext3/Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3;
    "
    
    
    
     