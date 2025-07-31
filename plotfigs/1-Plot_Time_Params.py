import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter 

# --- 全局样式设置 (保持不变) ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'
plt.style.use('seaborn-v0_8-whitegrid')

# BASE_COLOR = '#007ACC'  # 专业、清晰的蓝色
# BASE_COLOR = '#547DA1'  
# ALPHA_SOTA = 0.6       # 基线方法的透明度 (PLOP+NeST)
# ALPHA_OURS = 1.0       # 我们方法的透明度 (DFT*), 完全不透明
BASE_COLOR_SOTA = '#72AEC1'  # PLOP+NeST 的颜色
BASE_COLOR_OURS = '#F3CFE3'  # DFT* (Ours) 的颜色
ALPHA_SOTA = 0.6       # PLOP+NeST 的透明度
ALPHA_OURS = 0.6       # DFT* (Ours) 的透明度

# --- 数据定义 (保持不变) ---
datasets = ['VOC 10-1 (11 steps)', 'VOC 15-1 (6 steps)', 'VOC 15-5 (2 steps)', 'VOC 19-1 (2 steps)',
            'ADE 100-50 (2 steps)', 'ADE 100-10 (6 steps)', 'ADE 100-5 (11 steps)', 'ADE 50-50 (3 steps)']
width = 0.25
VOC_datas = {
    "Res101": {
        "VOC 10-1 (11 steps)": {"sota": [13962/60, 58035521/1e6], "ours": [4719/60, 257*5.5/1e6]},
        "VOC 15-1 (6 steps)": {"sota": [4444/60, 58035521/1e6], "ours": [1889/60, 257*3/1e6]},
        "VOC 15-5 (2 steps)": {"sota": [2878/60, 58036549/1e6], "ours": [885/60, 1285/1e6]},
        "VOC 19-1 (2 steps)": {"sota": [874/60, 58035521/1e6], "ours": [373/60, 257/1e6]},
        "ADE 100-50 (2 steps)": {"sota": [18500/60, 58074071/1e6], "ours": [6546/60, 12850/1e6]},
        "ADE 100-10 (6 steps)": {"sota": [27113/60, 58068931/1e6], "ours": [10353/60, 2570*3/1e6]},
        "ADE 100-5 (11 steps)": {"sota": [30592/60, 58068289/1e6], "ours": [12385/60, 1285*5.5/1e6]},
        "ADE 50-50 (3 steps)": {"sota": [39920/60, 58067646/1e6], "ours": [15299/60, 12850*1.5/1e6]},
    },
    "Swin-B": {
        "VOC 10-1 (11 steps)": {"sota": [13743/60, 94813561/1e6], "ours": [4660/60, 257*5.5/1e6]},
        "VOC 15-1 (6 steps)": {"sota": [4621/60, 94813561/1e6], "ours": [1707/60, 257*3/1e6]},
        "VOC 15-5 (2 steps)": {"sota": [3041/60, 94814589/1e6], "ours": [920/60, 1285/1e6]},
        "VOC 19-1 (2 steps)": {"sota": [870/60, 94813561/1e6], "ours": [380/60, 257/1e6]},
        "ADE 100-50 (2 steps)": {"sota": [18894/60, 94852111/1e6], "ours": [7705/60, 12850/1e6]},
        "ADE 100-10 (6 steps)": {"sota": [27369/60, 94846971/1e6], "ours": [12120/60, 2570*3/1e6]},
        "ADE 100-5 (11 steps)": {"sota": [30914/60, 94846329/1e6], "ours": [14091/60, 1285*5.5/1e6]},
        "ADE 50-50 (3 steps)": {"sota": [40298/60, 94845686/1e6], "ours": [18391/60, 12850*1.5/1e6]},
    },
}
# 4719/13962
# 1889/4444
# 885/2878
# 373/874
# 6546/18500
# 10353/27113
# 12385/30592
# 15299/39920
# 4660/13743
# 1707/4621
# 920/3041
# 380/870
# 7705/18894
# 12120/27369
# 14091/30914
# 18391/40298

# 257*5.5/58035521
# 257*3/58035521
# 1285/58036549
# 257/58035521
# 12850/58074071
# 2570*3/58068931
# 1285*5.5/58068289
# 12850*2/58067646
# 257*5.5/94813561
# 257*3/94813561
# 1285/94814589
# 257/94813561
# 12850/94852111
# 2570*3/94846971
# 1285*5.5/94846329
# 12850*1.5/94845686


# --- 辅助函数 (保持不变) ---
def format_scientific(value):
    s = "{:.2e}".format(value)
    base, exp = s.split('e')
    exp = int(exp)
    return rf"{float(base):.2f}$\times$10$^{{{exp}}}$"

# --- 绘图函数 (已修改) ---
def plot_subplot(ax, sota_data, ours_data, title=None):
    """绘制单个子图，已修正网格和刻度标签问题"""
    ax_right = ax.twinx()

    time_sota, param_sota = sota_data
    time_ours, param_ours = ours_data

    time_pos, param_pos = 0, 0.8

    # 绘制柱状图
    # ax.bar(time_pos - width/2, time_sota, width, color='#72AEC1')
    # ax.bar(time_pos + width/2, time_ours, width, color='#F3CFE3')
    # ax_right.bar(param_pos - width/2, param_sota, width, color='#72AEC1')
    # ax_right.bar(param_pos + width/2, param_ours, width, color='#F3CFE3')
    ax.bar(time_pos - width/2, time_sota, width, color=BASE_COLOR_SOTA, alpha=ALPHA_SOTA)
    ax.bar(time_pos + width/2, time_ours, width, color=BASE_COLOR_OURS, alpha=ALPHA_OURS)

    # 动态设置Y轴范围，增加一些边距以适应不同数据
    min_param = min(p for p in [param_sota, param_ours] if p > 0) # 忽略0值
    max_param = max(param_sota, param_ours)
    log_axis_bottom = min_param * 0.5 

    ax_right.bar(param_pos - width/2, param_sota, width, color=BASE_COLOR_SOTA, alpha=ALPHA_SOTA, bottom=log_axis_bottom)
    ax_right.bar(param_pos + width/2, param_ours, width, color=BASE_COLOR_OURS, alpha=ALPHA_OURS, bottom=log_axis_bottom)

    # 设置X轴
    ax.set_xticks([time_pos, param_pos])
    ax.set_xticklabels(['Time (min)', 'Params (M)'], fontsize=16)

    # --- Y轴设置 ---
    # 左侧Y轴 (Time)
    ax.set_ylabel("Time (min)", fontsize=16, weight='bold')
    max_time = max(time_sota, time_ours)
    ax.set_ylim(0, max_time * 1.2)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(False) # 关闭左侧Y轴网格

    # 右侧Y轴 (Params)
    ax_right.set_ylabel("Params (M)", fontsize=16, weight='bold')
    ax_right.set_yscale('log')
    # ax_right.set_ylim(1e-4, 1e4)
    ax_right.set_ylim(log_axis_bottom, max_param * 5)
    ax_right.tick_params(axis='y', labelsize=14)
    # 1. 强制关闭右侧Y轴的所有网格线 (主要和次要)
    ax_right.grid(False, which='both')
    # 2. 隐藏右侧Y轴的次要刻度标签 (例如 0.2, 0.4...)
    ax_right.yaxis.set_minor_formatter(NullFormatter())

    # 添加数值标签
    def add_labels(bars, axis, is_param=False, y_offset=None):
        if y_offset is None:
            y_offset = [3] * len(bars)  # 默认偏移量为3
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height < 1e-5: continue
            label = format_scientific(height) if is_param else f"{height:.1f}"
            axis.annotate(label, (bar.get_x() + bar.get_width() / 2, height),
                          textcoords="offset points", xytext=(0, y_offset[i]),
                          ha='center', va='bottom', fontsize=14, fontweight='bold')

    add_labels(ax.patches[:2], ax)
    add_labels(ax_right.patches, ax_right, is_param=True, y_offset=[3, 15])

    if title:
        ax.set_title(title, fontsize=18, pad=20)


# --- 主逻辑 (保持不变) ---
models_to_plot = ['Res101', 'Swin-B']

for model in models_to_plot:
    for dataset in datasets:
        fig_single, ax_single = plt.subplots(figsize=(8, 7), dpi=100)
        data = VOC_datas[model][dataset]

        plot_title = f'{model} Backbone\n{dataset}'
        plot_subplot(ax_single, data['sota'], data['ours'], title=plot_title)

        # legend_elements = [Patch(facecolor='#72AEC1', label='PLOP+NeST'),
        #                    Patch(facecolor='#F3CFE3', label='DFT* (Ours)')]
        legend_elements = [Patch(facecolor=BASE_COLOR_SOTA, alpha=ALPHA_SOTA, label='PLOP+NeST'),
                           Patch(facecolor=BASE_COLOR_OURS, alpha=ALPHA_OURS, label='DFT* (Ours)')]

        ax_single.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                         ncol=2, fontsize=18, frameon=False)

        sanitized_dataset_name = re.sub(r'[\\/*?:"<>|()\s]', '_', dataset).strip('_')
        base_filename = f"plot_{model}_{sanitized_dataset_name}"

        print(f"正在保存: {base_filename}.svg / .pdf")
        fig_single.savefig(f'/data/Code/CSS/Rethinking/plotfigs/Figure1/{base_filename}.svg', dpi=500, bbox_inches='tight')
        # fig_single.savefig(f'/data/Code/CSS/Rethinking/plotfigs/Figure1/{base_filename}.pdf', dpi=500, bbox_inches='tight')

        plt.close(fig_single)

print("\n所有子图已单独保存完成。")