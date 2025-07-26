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
TASK=100-5
NAME=FT-SwinB-FixBC-P-${LR}
METHOD=FT 
# relevant parameters are set here
results_dir=results_ep60
OPTIONS="--backbone swin_b --checkpoint /mnt/gpfs/renyong/CSS/checkpoints_ep60/step/ --fix_bachbone --fix_pre_cls --pre_allocate --task_num 11 --results_dir ${results_dir}"


SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=${results_dir}/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

# If you already trained the model for the first step, you can re-use those weights
# in order to skip this initial step --> faster iteration on your model
# Set this variable with the weights path
# FIRSTMODEL=/home/renyong/ECCV24_NeST/checkpoints/step/15-5s-voc_FT-fixbb-fixcls-preallocate_0.pth
# Then, for the first step, append those options:
# --ckpt ${FIRSTMODEL} --test
# And for the second step, this option:
# --step_ckpt ${FIRSTMODEL}

# NEW_FIRSTMODEL=/home/renyong/ECCV24_NeST/checkpoints/step/${TASK}-${DATASET}_${NAME}_0.pth
# cp ${FIRSTMODEL} ${NEW_FIRSTMODEL}

BATCH_SIZE=$((24 / NB_GPU))
INITIAL_EPOCHS=60
EPOCHS=60

if [ -f /mnt/gpfs/renyong/CSS/checkpoints_ep60/step/${TASK}-${DATASET}_FT-SwinB-FixBC-P-0.05_0.pth ]; then
    echo "Task 0 checkpoint already exists, skipping ..."
    if [ ! -f /mnt/gpfs/renyong/CSS/checkpoints_ep60/step/${TASK}-${DATASET}_${NAME}_0.pth ]; then
        echo "Copying task 0 checkpoint to new name ..."
        cp /mnt/gpfs/renyong/CSS/checkpoints_ep60/step/${TASK}-${DATASET}_FT-SwinB-FixBC-P-0.05_0.pth /mnt/gpfs/renyong/CSS/checkpoints_ep60/step/${TASK}-${DATASET}_${NAME}_0.pth
    fi
    CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.02 --epochs ${INITIAL_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --test
else
    echo "Task 0 checkpoint does not exist, running task 0 training ..."
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.02 --epochs ${INITIAL_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} 
fi

mid=`date +%s`
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 1 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} 
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 2 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 3 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 4 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 5 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 6 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 7 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 8 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 9 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 10 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}
python3 average_csv.py ${RESULTSFILE}

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"

runtime_continue_learning=$((end-mid))
echo "Run continue learning in ${runtime_continue_learning}s"