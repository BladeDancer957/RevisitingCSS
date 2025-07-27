#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))

if [ $# -lt 2 ]; then
    echo "Usage: $0 <GPU> <NB_GPU>"
    exit 1
fi

GPU=$1
NB_GPU=$2
LR=$3

DATA_ROOT=/data/Code/CSS/Rethinking/data

DATASET=ade
TASK=50
NAME=PLOP_NEST-SwinB-${LR}
METHOD=FT 
# relevant parameters are set here
results_dir=results_ep30_60
OPTIONS="--backbone swin_b --warm_up --warm_epochs 15 --warm_lr_scale 500 --unce_in_warm --checkpoint /mnt/gpfs/renyong/CSS/checkpoints_ep30_60/step/ --pod local --pod_factor 0.001 --pod_logits --pseudo entropy --threshold 0.001 --classif_adaptive_factor --init_balanced --results_dir ${results_dir}"



SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=${results_dir}/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

# If you already trained the model for the first step, you can re-use those weights
# in order to skip this initial step --> faster iteration on your model
# Set this variable with the weights path
# FIRSTMODEL=/path/to/my/first/weights
# Then, for the first step, append those options:
# --ckpt ${FIRSTMODEL} --test
# And for the second step, this option:
# --step_ckpt ${FIRSTMODEL}

BATCH_SIZE=$((24 / NB_GPU))
INITIAL_EPOCHS=30
EPOCHS=60

CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.02 --epochs ${INITIAL_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}

mid=`date +%s`
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 1 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.00001, \"type\": \"local\"}}}"
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 2 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.00001, \"type\": \"local\"}}}"
python3 average_csv.py ${RESULTSFILE}

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"

runtime_continue_learning=$((end-mid))
echo "Run continue learning in ${runtime_continue_learning}s"
