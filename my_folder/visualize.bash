singularity exec --nv \
    --overlay /scratch/wz3008/data/movi_a_info.sqf:ro \
    --overlay /scratch/wz3008/overlay-50G-10M-dl.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /ext3/env.sh; \
    conda activate oclf; \
    cd /scratch/wz3008/new_SlotAttn/object-centric-learning-framework/; \
    export PATH="/ext3/share/pipx/venvs/poetry/bin:$PATH"; \
    export PYTHONPATH="$(pwd):$PYTHONPATH"; \
    export DATASET_PREFIX=/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/scripts/datasets/outputs; \
    poetry run python my_folder/train.py --train_percent 0.001 --visualize \
        --checkpoint_path '/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/outputs/slot_attention/movi_a/2026-01-29_07-48-41';"
        

# '/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/outputs/slot_attention/movi_c/2026-01-28_00-29-00/'


# poetry run python my_folder/train.py --train_percent 0.001 --visualize --checkpoint_path '/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/outputs/slot_attention/movi_a/2026-01-29_07-48-41/' --use_original_data