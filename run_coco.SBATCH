#!/bin/bash
#SBATCH --output=/scratch/wz3008/SlotAttn/slurm_outputs/%x_%j.out
#SBATCH --error=/scratch/wz3008/SlotAttn/slurm_outputs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wz3008@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=oclf_movi_c
#SBATCH --account torch_pr_36_mren

srun singularity exec --nv \
    --overlay /scratch/wz3008/overlay-50G-10M-dl.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /ext3/env.sh; \
    conda activate oclf; \
    cd /scratch/wz3008/new_SlotAttn/object-centric-learning-framework/; \
    export PATH="/ext3/share/pipx/venvs/poetry/bin:$PATH"; \
    export PYTHONPATH="$(pwd):$PYTHONPATH"; \
    export DATASET_PREFIX=/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/scripts/datasets/outputs; \
    poetry run ocl_train +experiment=slot_attention/movi_c trainer.devices=1"