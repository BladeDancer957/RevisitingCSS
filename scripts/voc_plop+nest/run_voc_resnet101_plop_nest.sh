unique_identifier=$(date +"%Y%m%d_%H%M%S")
mkdir -p shell_logs_bl
GPU_LST=(
    0
    1
    2
    3
)
NB_GPU=1
LR=0.001
SubDir="voc_plop+nest"
FileNames=(
    "plop+ours_resnet101_10-1.sh"
    "plop+ours_resnet101_15-1.sh"
    "plop+ours_resnet101_15-5.sh"
    "plop+ours_resnet101_19-1.sh"
)
for i in "${!GPU_LST[@]}";
do
    echo "Start running on GPU ${GPU_LST[i]} for ${FileNames[i]}"
    nohup bash /data/Code/CSS/Rethinking/scripts/${SubDir}/${FileNames[i]} ${GPU_LST[i]} ${NB_GPU} ${LR} > shell_logs_bl/${unique_identifier}_${FileNames[i]}_${LR}.log 2>&1 &
done