unique_identifier=$(date +"%Y%m%d_%H%M%S")
mkdir -p shell_logs_ep30_60
GPU_LST=(
    4
    5
    6
    7
)
NB_GPU=1
LR=0.1
SubDir="ep30_60/ade_resnet101_FixBC-P"
FileNames=(
    "ft_resnet101_50-50-FixBC-P.sh"
    "ft_resnet101_100-5-FixBC-P.sh"
    "ft_resnet101_100-10-FixBC-P.sh"
    "ft_resnet101_100-50-FixBC-P.sh"
)
for i in "${!GPU_LST[@]}";
do
    echo "Start running on GPU ${GPU_LST[i]} for ${FileNames[i]}"
    nohup bash /data/Code/CSS/Rethinking/scripts/${SubDir}/${FileNames[i]} ${GPU_LST[i]} ${NB_GPU} ${LR} > shell_logs_ep30_60/${unique_identifier}_${FileNames[i]}_${LR}.log 2>&1 &
done