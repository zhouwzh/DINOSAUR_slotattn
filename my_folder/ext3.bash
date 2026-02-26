#!/bin/bash

cd /ext3;

rm -rf miniforge3/;

wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh;

echo '---Miniforge3-Linux-x86_64.sh downloaded---'

bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3;

ecoh '---miniforge3 created---'

rm -rf Miniforge3-Linux-x86_64.sh






#singularity exec --nv --fakeroot \
#    --overlay /scratch/wz3008/overlay-50G-10M-oclf.ext3:ro \
#    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
#    -c "source /ext3/env.sh; \
#    conda activate oclf; \
#    cd /scratch/wz3008/new_SlotAttn/object-centric-learning-framework/; \
#   export PATH="/ext3/share/pipx/venvs/poetry/bin:$PATH"; \
#   export PYTHONPATH="$(pwd):$PYTHONPATH"; \
#   export DATASET_PREFIX=/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/scripts/datasets/outputs; \
#    poetry run ocl_train +experiment=slot_attention/movi_c trainer.devices=1"
#    
# fix bug at ocl/metrics/masks.py line 335
# fix bug DATASET_PREFIX should be absolute path
    
    # used in install
    #export HOME=/ext3/home/; \
    #export XDG_CACHE_HOME=/ext3/cache; \
    #export TMPDIR=/ext3/tmp; \
    #export PIP_CACHE_DIR=/ext3/pip-cache; \
    #export HYDRA_FULL_ERROR=1; \
    