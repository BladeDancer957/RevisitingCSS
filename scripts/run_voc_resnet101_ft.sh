unique_identifier=$(date +"%Y%m%d_%H%M%S")
global_name="voc_resnet101_ft"
mkdir -p shell_logs
GPU_LST=(
    0
    1
    2
    3
)
FileNames=(
    "ft_resnet101_10-1"
    "ft_resnet101_15-1"
    "ft_resnet101_15-5"
    "ft_resnet101_19-1"
)
for i in "${!GPU_LST[@]}";
do
    echo "Start running on GPU $i"
    nohup bash /home/renyong/ECCV24_NeST/scripts/voc/${FileNames[i]}.sh ${GPU_LST[i]} 1 > shell_logs/${unique_identifier}_${FileNames[i]}.log 2>&1 &
done