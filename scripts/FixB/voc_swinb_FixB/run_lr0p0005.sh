unique_identifier=$(date +"%Y%m%d_%H%M%S")
mkdir -p shell_logs_fixb
GPU_LST=(
    0,1
    2,3
    4,5
    6,7
)
NB_GPU=2
LR=0.0005
SubDir="FixB/voc_swinb_FixB"
FileNames=(
    "ft_swinb_10-1-FixB.sh"
    "ft_swinb_15-1-FixB.sh"
    "ft_swinb_15-5-FixB.sh"
    "ft_swinb_19-1-FixB.sh"
)
for i in "${!GPU_LST[@]}";
do
    echo "Start running on GPU ${GPU_LST[i]} for ${FileNames[i]}"
    nohup bash /data/Code/CSS/Rethinking/scripts/${SubDir}/${FileNames[i]} ${GPU_LST[i]} ${NB_GPU} ${LR} > shell_logs_fixb/${unique_identifier}_${FileNames[i]}_${LR}.log 2>&1 &
done