unique_identifier=$(date +"%Y%m%d_%H%M%S")
mkdir -p shell_logs
GPU_LST=(
    4
    5
    6
    7
)
NB_GPU=1
LR=0.08
SubDir="ade_resnet101_FixBC-P"
FileNames=(
    "ft_resnet101_50-50-FixBC-P.sh"
    "ft_resnet101_100-5-FixBC-P.sh"
    "ft_resnet101_100-10-FixBC-P.sh"
    "ft_resnet101_100-50-FixBC-P.sh"
)
for i in "${!GPU_LST[@]}";
do
    echo "Start running on GPU ${GPU_LST[i]} for ${FileNames[i]}"
    nohup bash /data/Code/CSS/Rethinking/scripts/${SubDir}/${FileNames[i]} ${GPU_LST[i]} ${NB_GPU} ${LR} > shell_logs/${unique_identifier}_${FileNames[i]}_${LR}.log 2>&1 &
done