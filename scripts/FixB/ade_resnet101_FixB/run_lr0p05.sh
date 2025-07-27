unique_identifier=$(date +"%Y%m%d_%H%M%S")
mkdir -p shell_logs_fixb
GPU_LST=(
    0
    1
    2
    3
)
NB_GPU=1
LR=0.05
SubDir="FixB/ade_resnet101_FixB"
FileNames=(
    "ft_resnet101_50-50-FixB.sh"
    "ft_resnet101_100-5-FixB.sh"
    "ft_resnet101_100-10-FixB.sh"
    "ft_resnet101_100-50-FixB.sh"
)
for i in "${!GPU_LST[@]}";
do
    echo "Start running on GPU ${GPU_LST[i]} for ${FileNames[i]}"
    nohup bash /data/Code/CSS/Rethinking/scripts/${SubDir}/${FileNames[i]} ${GPU_LST[i]} ${NB_GPU} ${LR} > shell_logs_fixb/${unique_identifier}_${FileNames[i]}_${LR}.log 2>&1 &
done