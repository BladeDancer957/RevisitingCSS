#!/bin/bash

set -euo pipefail

# Configuration
readonly START_TIME=$(date +%s)
readonly START_DATE=$(date '+%Y-%m-%d')
readonly BASE_PORT=$((9000 + RANDOM % 1000))

# Validate arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <GPU> <NB_GPU>"
    exit 1
fi

readonly GPU=$1
readonly NB_GPU=$2

# Constants
readonly DATA_ROOT="/data/Code/CSS/Rethinking/data/PascalVOC12"
readonly DATASET="ade"
readonly TASK="100-5-probing"
readonly NAME="FT-SwinB-FixB-0.1"
readonly METHOD="FT"
readonly OPTIONS="--backbone swin_b --checkpoint /mnt/gpfs/renyong/CSS/checkpoints_probing/step/ --fix_backbone --test"
readonly SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"
readonly BATCH_SIZE=$((24 / NB_GPU))
readonly EPOCHS=30

echo "Starting ${SCREENNAME}"
echo "Batch size: ${BATCH_SIZE}"

# Function to run training for a specific step
run_training_step() {
    local step=$1
    local port=$((BASE_PORT + (step - 1) * 11))
    
    CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch \
        --master_port ${port} \
        --nproc_per_node=${NB_GPU} \
        probing_cls_class_embedding_get.py \
        --date ${START_DATE} \
        --data_root ${DATA_ROOT} \
        --overlap \
        --batch_size ${BATCH_SIZE} \
        --dataset ${DATASET} \
        --name ${NAME} \
        --task ${TASK} \
        --step ${step} \
        --lr 0.001 \
        --epochs ${EPOCHS} \
        --method ${METHOD} \
        --opt_level O1 \
        ${OPTIONS}
}

# Run training for all steps
for step in {1..11}; do
    echo "Starting training for step ${step}"
    run_training_step ${step}
done

echo "Completed ${SCREENNAME}"