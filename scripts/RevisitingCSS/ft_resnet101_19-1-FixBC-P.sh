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
LR=0.005
checkpoint_dir=checkpoints/step
results_dir=results

DATA_ROOT=/data/Code/CSS/Rethinking/data/PascalVOC12

DATASET=voc
TASK=19-1
NAME=FT-FixBC-P
METHOD=FT 
# relevant parameters are set here
OPTIONS="--backbone resnet101 --checkpoint ${checkpoint_dir} --fix_backbone --fix_pre_cls --pre_allocate --task_num 2 --results_dir ${results_dir}"


SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=${results_dir}/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

BATCH_SIZE=$((24 / NB_GPU))
INITIAL_EPOCHS=30
EPOCHS=30

CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.02 --epochs ${INITIAL_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} 

mid=`date +%s`
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 1 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} 
python3 average_csv.py ${RESULTSFILE}

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"

runtime_continue_learning=$((end-mid))
echo "Run continue learning in ${runtime_continue_learning}s"
