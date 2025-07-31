unique_identifier=$(date +"%Y%m%d_%H%M%S")
mkdir -p shell_logs_bl
GPU_LST=(
    0
    1
    2,3
    4,5
)
NB_GPU=(
    1
    1
    2
    2
)
LR=0.001
SubDir="voc_plop+nest_ep30-10"
FileNames=(
    "plop+ours_resnet101_10-1.sh"
    "plop+ours_resnet101_15-1.sh"
    "plop+ours_swin_b_10-1.sh"
    "plop+ours_swin_b_15-1.sh"
)
for i in "${!GPU_LST[@]}";
do
    echo "Start running on GPU ${GPU_LST[i]} for ${FileNames[i]}"
    nohup bash /data/Code/CSS/Rethinking/scripts/${SubDir}/${FileNames[i]} ${GPU_LST[i]} ${NB_GPU[i]} ${LR} > shell_logs_bl/${unique_identifier}_${FileNames[i]}_ep30_10_${LR}.log 2>&1 &
done