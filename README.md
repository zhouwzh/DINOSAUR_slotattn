This repo is adapted from amazon's [object-centric-learning-framework](https://github.com/amazon-science/object-centric-learning-framework), git clone this repo or just git clone object-centric-learning-framework add `my_folder` to it.

Modify to OCLF: fix bug at ocl/metrics/masks.py line 335.

To run oclf training: `sbatch run.SBATCH` or `poetry run ocl_train +experiment=slot_attention/movi_a`.

To run my training: `sbatch my_folder/run_slot_1.SBATCH` or `poetry run python my_folder/train.py --train_percent 0.001`.

