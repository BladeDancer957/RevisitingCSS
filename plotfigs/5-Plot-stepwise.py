import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import csv

# 设置全局字体为Arial并加粗
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'
plt.style.use('seaborn-v0_8-whitegrid')

save_base_path = "/data/Code/CSS/Rethinking/plotfigs/Figure6"
os.makedirs(save_base_path, exist_ok=True)


# get meta data

datasets = ['voc', 'ade']
tasks_for_voc = ['10-1', '15-5s']
tasks_for_ade = ['100-5', '100-10']
backbones = ['resnet101', 'swinb']

plop_nest_results_list = []
fixbc_results_list = []
fixbc_p_results_list = []

for dataset in datasets:
    tasks = tasks_for_voc if dataset == 'voc' else tasks_for_ade
    for task in tasks:
        for backbone in backbones:
            BB = '-SwinB' if backbone == 'swinb' else ''
            LR = 0.05 if dataset == 'ade' else 0.0005
            fixbc_path = f"/data/Code/CSS/Rethinking/results/2025-07-27_{dataset}_{task}_FT{BB}-FixBC-{LR}.csv"
            LR = 0.1 if dataset == 'ade' else 0.0005
            if dataset == 'ade':
                # /data/Code/CSS/Rethinking/results_ep30_60/2025-07-27_ade_100-10_PLOP_NEST-0.001.csv
                plop_nest_path = f"/data/Code/CSS/Rethinking/results_ep30_60/2025-07-27_ade_{task}_PLOP_NEST{BB}-0.001.csv"
                if task == '100-5':
                    fixbc_p_path = f"/data/Code/CSS/Rethinking/results_ep30_60/2025-07-26_ade_{task}_FT{BB}-FixBC-P-0.05.csv"
                else:
                    fixbc_p_path = f"/data/Code/CSS/Rethinking/results_ep30_60/2025-07-27_ade_{task}_FT{BB}-FixBC-P-{LR}.csv"
            elif dataset == 'voc':
                # /data/Code/CSS/Rethinking/results/2025-07-25_voc_10-1_PLOP_NEST-0.001.csv
                if task == '10-1':
                    plop_nest_path = f"/data/Code/CSS/Rethinking/results_ep30_10/2025-07-31_voc_{task}_PLOP_NEST{BB}-0.001.csv"
                elif task == '15-5s':
                    plop_nest_path = f"/data/Code/CSS/Rethinking/results/2025-07-31_voc_{task}_PLOP_NEST{BB}-0.001.csv"
                # plop_nest_path = f"/data/Code/CSS/Rethinking/results_plop_nest/2025-07-25_voc_{task}_PLOP_NEST{BB}-0.001.csv"
                fixbc_p_path = f"/data/Code/CSS/Rethinking/results/2025-07-25_voc_{task}_FT{BB}-FixBC-P-{LR}.csv"

            # if each file is exist
            if os.path.exists(plop_nest_path):
                print(f"Loading {plop_nest_path}")
                with open(plop_nest_path, 'r') as f:
                    reader = csv.reader(f)
                    # load last of each row
                    miou_step_wise = []
                    for row in reader:
                        if row:
                            miou_step_wise.append(float(row[-1]))
                plop_nest_results_list.append(miou_step_wise)
                # import pdb; pdb.set_trace()

            else:
                raise FileNotFoundError(f"File not found: {plop_nest_path}")
            if os.path.exists(fixbc_path):
                print(f"Loading {fixbc_path}")
                with open(fixbc_path, 'r') as f:
                    reader = csv.reader(f)
                    # load last of each row
                    miou_step_wise = []
                    for row in reader:
                        if row:
                            miou_step_wise.append(float(row[-1]))
                fixbc_results_list.append(miou_step_wise)
            else:
                raise FileNotFoundError(f"File not found: {fixbc_path}")
            if os.path.exists(fixbc_p_path):
                print(f"Loading {fixbc_p_path}")
                with open(fixbc_p_path, 'r') as f:
                    reader = csv.reader(f)
                    # load last of each row
                    miou_step_wise = []
                    for row in reader:
                        if row:
                            miou_step_wise.append(float(row[-1]))
                fixbc_p_results_list.append(miou_step_wise)
            else:
                raise FileNotFoundError(f"File not found: {fixbc_p_path}")
            
colors = {'fix_bc_p': '#D5818B', 'plop_nest': '#9EB8E3', 'fix_bc': "#F4C4848B"}
marker = {'fix_bc_p': '*', 'plop_nest': 'o', 'fix_bc': 'd'}
mark_size = {'fix_bc_p': 11, 'plop_nest': 8, 'fix_bc': 9}

for i in range(len(plop_nest_results_list)):
    dataset = datasets[0] if i < len(plop_nest_results_list) // 2 else datasets[1]
    tasks_list = tasks_for_voc if dataset == 'voc' else tasks_for_ade 
    offset = 0 if dataset == 'voc' else len(plop_nest_results_list) // 2
    task = tasks_list[0] if i - offset < len(plop_nest_results_list) // 4 else tasks_list[1]
    backbone = backbones[0] if i % 2 == 0 else backbones[1] 
    plop_nest_result = plop_nest_results_list[i]
    fixbc_result = fixbc_results_list[i]
    fixbc_p_result = fixbc_p_results_list[i]
    # plot the results
    plt.figure(figsize=(8, 6))
    steps = np.arange(len(plop_nest_result)) + 1
    int_steps = np.arange(1, len(plop_nest_result) + 1)  # 从1开始
    plt.xticks(int_steps, [str(int(x)) for x in int_steps], fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(steps, plop_nest_result, marker=marker['plop_nest'], color=colors['plop_nest'], linewidth=2, markersize=mark_size['plop_nest'], label='PLOP-NEST')
    plt.plot(steps, fixbc_result, marker=marker['fix_bc'], color=colors['fix_bc'], linewidth=2, markersize=mark_size['fix_bc'], label='FixBC')
    plt.plot(steps, fixbc_p_result, marker=marker['fix_bc_p'], color=colors['fix_bc_p'], linewidth=2, markersize=mark_size['fix_bc_p'], label='FixBC-P')

    # 在 FixBC-P 的每个点上添加数值标签（保留1位小数）
    # for x, y in zip(steps, fixbc_p_result):
    # for x, y in zip(steps, fixbc_result):
    # for x, y in zip(steps, plop_nest_result):
    #     plt.text(x, y, f"{y:.3f}", 
    #              ha='center', va='bottom',  # 水平居中，垂直在点上方
    #              fontsize=12, fontweight='bold', 
    #              color=colors['fix_bc_p'])

    plt.title(f"{dataset.upper()} {task} {backbone.upper()}", fontsize=18, pad=10)
    plt.xlabel('Step', fontsize=18)
    plt.ylabel('mIoU', fontsize=18)
    # plt.xticks(steps, [str(int(x)) for x in steps], fontsize=14)
    # plt.yticks(fontsize=14)
    plt.grid(False)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    plt.gca().yaxis.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    # save the figure
    save_path = os.path.join(save_base_path, f"{dataset}_{task}_{backbone}.svg")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭当前图表，释放内存
print("所有图表已保存完成！")
