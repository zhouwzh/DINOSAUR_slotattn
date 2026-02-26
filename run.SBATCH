#!/bin/bash
#SBATCH --output=/scratch/wz3008/SlotAttn/slurm_outputs/%x_%j.out
#SBATCH --error=/scratch/wz3008/SlotAttn/slurm_outputs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wz3008@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=3
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:3
#SBATCH --job-name=continue_oclf_movi_a
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
    poetry run ocl_train +experiment=slot_attention/movi_a trainer.devices=1"
    
#poetry run ocl_train \
#        +experiment=/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/outputs/slot_attention/movi_a/2026-01-29_07-48-41/config/config.yaml \
#        trainer.devices=1
    
# fix bug at ocl/metrics/masks.py line 335
# fix bug DATASET_PREFIX should be absolute path
    
    # used in install
    export HOME=/ext3/home/; \
    export XDG_CACHE_HOME=/ext3/cache; \
    export TMPDIR=/ext3/tmp; \
    export PIP_CACHE_DIR=/ext3/pip-cache; \
    export HYDRA_FULL_ERROR=1; \
    
# process jincheng: independent space, (DDP: each gpu has own process)
# thread xiancheng: share space, cost less, 
# --ntasks-per-node: control process tasks

# srun: process all process simutanously



# DDP requires four processes, 

# poetry run python
#poetry run ocl_train +experiment=slot_attention/movi_c \
#    +load_checkpoint='/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/outputs/slot_attention/movi_c/2026-01-28_00-29-00/lightning_logs/version_0/checkpoints/epoch=23-step=176677.ckpt' \
#    trainer.devices=1


#configs/experiment/slot_attention/_base_large.yaml
#configs/experiment/slot_attention/_preprocessing_movi.yaml
#ocl/decoding.py
#configs/experiment/slot_attention/_metrics_clevr.yaml


########### About Python and Poetry ###############################
#which python
#python -c "import sys; print(sys.executable)"

#poetry run which python
#poetry run python -c "import sys; print(sys.executable)"


