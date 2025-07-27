unique_identifier=$(date +"%Y%m%d_%H%M%S")
mkdir -p shell_logs_fixbc
GPU_LST=(
    4
    5
    6
    7
)
NB_GPU=1
LR=0.0005
SubDir="FixBC/voc_resnet101_FixBC"
FileNames=(
    "ft_resnet101_10-1-FixBC.sh"
    "ft_resnet101_15-1-FixBC.sh"
    "ft_resnet101_15-5-FixBC.sh"
    "ft_resnet101_19-1-FixBC.sh"
)
for i in "${!GPU_LST[@]}";
do
    echo "Start running on GPU ${GPU_LST[i]} for ${FileNames[i]}"
    nohup bash /data/Code/CSS/Rethinking/scripts/${SubDir}/${FileNames[i]} ${GPU_LST[i]} ${NB_GPU} ${LR} > shell_logs_fixbc/${unique_identifier}_${FileNames[i]}_${LR}.log 2>&1 &
done