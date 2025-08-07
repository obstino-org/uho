singularity exec --nv --bind ./data:/mnt whisper.sif bash -c "cd /mnt && python3 train.py base"
singularity exec --nv --bind ./data:/mnt whisper.sif bash -c "cd /mnt && python3 train.py tiny"
