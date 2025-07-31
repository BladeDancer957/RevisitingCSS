import matplotlib.pyplot as plt
import numpy as np
import os

# --- 全局样式设置 ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'
plt.style.use('seaborn-v0_8-whitegrid')

BASE_COLOR_SOTA = '#72AEC1'  # PLOP+NeST 的颜色
BASE_COLOR_OURS = '#F3CFE3'  # DFT* (Ours) 的颜色
ALPHA_SOTA = 0.6       # PLOP+NeST 的透明度
ALPHA_OURS = 0.6       # DFT* (Ours) 的透明度

# --- 数据定义 ---
# 1. 将数据组织到字典中，方便循环调用
all_data = {
    "ResNet101": {
        "sota": [36.9, 13962/60, 58035521/1e6],
        "ours": [44.4, 4719/60, 257*5.5/1e6]
    },
    "Swin-B": {
        "sota": [47.2, 13743/60, 94813561/1e6],
        "ours": [58.0, 4660/60, 257*5.5/1e6]
    }
}
metrics = ['mIoU (%)', 'Time (min)', 'Params (M)']
width = 0.35

# --- 辅助函数 ---
def format_scientific(value):
    """将科学计数法转换为LaTeX上标格式"""
    s = f"{value:.2e}"
    base, exp = s.split('e')
    exp = int(exp)
    return rf"{float(base):.2f}$\times$10$^{{{exp}}}$"

# --- 核心绘图函数 (已修改，用于绘制单个图表) ---
def plot_bars(ax, sota_data, ours_data):
    """在给定的axes上绘制所有柱状图、标签和图例"""
    ax_right = ax.twinx()  # 创建右侧Y轴

    # --- 【推荐】定义Y轴下限为一个变量，方便复用 ---
    log_axis_bottom = 1e-3

    # 准备数据
    left_data_sota = sota_data[:2]
    left_data_ours = ours_data[:2]
    right_data_sota = sota_data[2]
    right_data_ours = ours_data[2]

    # 绘制左侧Y轴的柱状图 (mIoU, Time)
    x_left = [0, 1]
    bars1 = ax.bar(np.array(x_left) - width/2, left_data_sota, width,
                   label='PLOP+NeST', color=BASE_COLOR_SOTA, alpha=ALPHA_SOTA)
    bars2 = ax.bar(np.array(x_left) + width/2, left_data_ours, width,
                   label='DFT* (Ours)', color=BASE_COLOR_OURS, alpha=ALPHA_OURS)

    # 绘制右侧Y轴的柱状图 (Params)
    x_right = 2
    bars3 = ax_right.bar(x_right - width/2, right_data_sota, width, color=BASE_COLOR_SOTA, alpha=ALPHA_SOTA, bottom=log_axis_bottom)
    bars4 = ax_right.bar(x_right + width/2, right_data_ours, width, color=BASE_COLOR_OURS, alpha=ALPHA_OURS, bottom=log_axis_bottom)

    # --- 坐标轴设置 ---
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(metrics, fontsize=20)

    # ax.set_xlim(-width, 2 + width) # 从第一个柱子的左边缘到最后一个柱子的右边缘
    
    # 2. 将Y轴标签直接设置在坐标轴上
    ax.set_ylabel('mIoU (%) & Time (min)', fontsize=20)
    ax_right.set_ylabel('Params (M)', fontsize=20, rotation=-90, va="bottom")

    ax.tick_params(axis='y', labelsize=16)
    # ax.set_ylim(0, max(max(left_data_sota), max(left_data_ours)) * 1.5) # 动态Y轴范围
    ax.set_ylim(0, 300)
    ax.grid(True, linestyle='--', alpha=0.6, which='both', zorder=0)
    # ax.grid(False)

    ax_right.set_yscale('log')
    ax_right.tick_params(axis='y', labelsize=16)
    ax_right.set_ylim(log_axis_bottom, 1e3) 
    # ax_right.set_ylim(1e-3, 1e3) # 调整范围以更好地显示数据
    ax_right.grid(False)

    # --- 数值标签函数 ---
    def add_labels(bars, axis, is_param=False, y_offset=5):
        for bar in bars:
            height = bar.get_height()
            label = format_scientific(height) if is_param else f'{height:.1f}'
            axis.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, y_offset), # 标签在柱顶上方的偏移量
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=16, fontweight='bold'
            )

    # 添加标签
    add_labels(bars1, ax)
    add_labels(bars2, ax)
    add_labels(bars3, ax_right, is_param=True)
    add_labels(bars4, ax_right, is_param=True, y_offset=15)
    
    # 3. 将图例直接添加到子图中
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=18, frameon=False)


# --- 主执行逻辑 ---
output_dir = "/data/Code/CSS/Rethinking/plotfigs/Figure2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 4. 循环遍历每个模型，创建并保存独立的图表
for model_name, data in all_data.items():
    # 5. 为每个模型创建一个新的、独立的画布
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    # 设置标题
    ax.set_title(model_name, fontsize=24, pad=20)

    # 在当前画布上绘图
    plot_bars(ax, data['sota'], data['ours'])

    # 调整布局以确保所有元素都可见
    plt.tight_layout(rect=[0, 0.1, 1, 1]) # 留出底部空间给图例

    # 6. 保存当前画布为独立文件
    base_filename = f"intro_{model_name}"
    output_path = os.path.join(output_dir, base_filename)
    print(f"正在保存: {output_path}.pdf / .svg")
    # fig.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_path}.svg', dpi=300, bbox_inches='tight')

    # 7. 关闭当前画布，释放内存
    plt.close(fig)

print(f"\n所有独立的子图已保存至 '{output_dir}' 文件夹。")