singularity exec --nv \
    --overlay /scratch/wz3008/data/saycam/5fps.sqf:ro \
    --overlay /scratch/yy2694/data/saycam/saycam_transcript_frames/saycam_transcript_5fps.sqf:ro \
    --overlay /scratch/wz3008/overlay-50G-10M-metaslot.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash