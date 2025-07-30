unique_identifier=$(date +"%Y%m%d_%H%M%S")
mkdir -p shell_logs_proto
GPU_LST=(
    # 0
    # 1
    2
    3
    # 4
    # 5
    6
    7
)
NB_GPU=1
SubDir="proto"
FileNames=(
    # "ft_resnet101_10-1_proto"
    # "ft_resnet101_15-1_proto"
    "ft_resnet101_100-5_proto"
    "ft_resnet101_100-10_proto"
    # "ft_swinb_10-1_proto"
    # "ft_swinb_15-1_proto"
    "ft_swinb_100-5_proto"
    "ft_swinb_100-10_proto"
)
for i in "${!GPU_LST[@]}";
do
    echo "Start running on GPU ${GPU_LST[i]} for ${FileNames[i]}"
    nohup bash /data/Code/CSS/Rethinking/scripts/${SubDir}/${FileNames[i]}.sh ${GPU_LST[i]} ${NB_GPU} > shell_logs_proto/${unique_identifier}_${FileNames[i]}.log 2>&1 &
done