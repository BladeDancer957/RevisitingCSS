#!/usr/bin/env bash

# --- 安全设置 ---
# -e: 如果命令失败，立即退出
# -u: 将未设置的变量视为错误
# -o pipefail: 如果管道中的任何命令失败，则整个管道失败
set -euo pipefail

readonly DATASET="ade"
readonly NUM_STEPS=7
readonly TASK="100-10"
readonly NAME="FT-FixB"
readonly METHOD="FT"
readonly BACKBONE="resnet101"
readonly LR="0.1"
readonly EPOCHS=30
readonly BATCH_SIZE=24
readonly NB_GPU_PER_NODE=1

readonly DATA_ROOT="/data/Code/CSS/Rethinking/data"
readonly CHECKPOINT_BASE_DIR="/mnt/gpfs/renyong/CSS/checkpoints/step"
readonly LOG_DIR="shell_logs_probing"
readonly START_PORT=11000 

readonly GPUS=(0 1 2 3 4 5 6 7)
readonly NUM_GPUS=${#GPUS[@]}

mkdir -p "${LOG_DIR}"
readonly START_DATE=$(date '+%Y-%m-%d')
echo "🚀 开始为数据集 '${DATASET}' 启动 ${NUM_STEPS} 个探测任务..."
echo "日志将保存在 ./${LOG_DIR} 目录下。"


for step in $(seq 0 $((NUM_STEPS - 1))); do
    gpu_id=${GPUS[step % NUM_GPUS]}
    master_port=$((START_PORT + step * 10))
    # 检测端口是否被占用
    while lsof -i :${master_port} >/dev/null; do
        echo "⚠️  端口 ${master_port} 已被占用，尝试下一个端口..."
        master_port=$((master_port + 1))
    done
    echo "使用 GPU ${gpu_id} 和端口 ${master_port} 进行 Step ${step} 的任务..."
    log_file="${LOG_DIR}/$(date +"%Y%m%d_%H%M%S")_${DATASET}_${TASK}-probing_${BACKBONE}_step_${step}.log"

    cmd_args=(
        "-m" "torch.distributed.launch" "--master_port" "${master_port}" "--nproc_per_node=${NB_GPU_PER_NODE}"
        "run_probing.py"
        "--date" "${START_DATE}"
        "--data_root" "${DATA_ROOT}"
        "--dataset" "${DATASET}"
        "--name" "${NAME}-${LR}"
        "--task" "${TASK}-probing" # 附加 'probing'
        "--step" "${step}"
        "--lr" "${LR}"
        "--epochs" "${EPOCHS}"
        "--method" "${METHOD}"
        "--batch_size" "${BATCH_SIZE}"
        "--backbone" "${BACKBONE}"
        "--checkpoint" "/mnt/gpfs/renyong/CSS/checkpoints_probing/step/"
        "--fix_backbone" # 注意：原始脚本中可能是拼写错误 (bachbone -> backbone)
        "--opt_level" "O1"
        "--overlap"
        "--results_dir" "results_probing"
    )

    # 从 step 1 开始，需要加载上一步的模型
    if [ "${step}" -gt 0 ]; then
        sleep 5
        prev_step=$((step - 1))
        checkpoint_path="${CHECKPOINT_BASE_DIR}/${TASK}-${DATASET}_${METHOD}-0.001_${prev_step}.pth"

        if [ ! -f "${checkpoint_path}" ]; then
            echo "⚠️  警告: 找不到 Step ${step} 需要的检查点文件: ${checkpoint_path}"
            echo "跳过此任务..."
            continue
        fi
        cmd_args+=("--step_ckpt" "${checkpoint_path}")
    fi

    echo "  -> 启动 Step ${step}: GPU=${gpu_id}, Port=${master_port}"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} nohup python3 "${cmd_args[@]}" > "${log_file}" 2>&1 &

    # step 0
    if [ "${step}" -eq 0 ]; then
        sleep 100
    fi

done

echo "✅ 所有 ${NUM_STEPS} 个任务已在后台启动。"