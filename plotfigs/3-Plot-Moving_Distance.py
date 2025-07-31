import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.path.append("/data/Code/CSS/Rethinking")  # 添加上级目录到路径
import tasks
import os

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'
plt.style.use('seaborn-v0_8-whitegrid')

para_list = []

for dataset in ['voc', 'ade']:
    if dataset == 'voc':
        for task in ['10-1', '15-5s']:
            for backbone in ['resnet101', 'swin_b']:
                for probing in [True, False]:
                    para_list.append([dataset, task, backbone, probing])
    elif dataset == 'ade':
        for task in ['100-5', '100-10']:
            for backbone in ['resnet101', 'swin_b']:
                for probing in [True, False]:
                    para_list.append([dataset, task, backbone, probing])

# idx is in 0-15
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 获取命令行参数或默认值0

# dataset='voc' # or 'ade'
# task='10-1' # or '15-5s', '100-5', '100-10'
# backbone = "resnet101" # or "swin_b"
# probing = True # False

dataset, task_name, backbone, probing = para_list[idx]  

if dataset == 'voc':
    lr = 0.001
elif dataset == 'ade':
    lr = 0.1

if backbone == "resnet101":
    BB="FT"
elif backbone == "swin_b":
    BB="FT-SwinB"

if probing:
    flag = 'Probing'
else:
    flag = 'Observed'

tasks_dict = tasks.get_task_dict(dataset, task_name)
step_num = len(tasks_dict)
all_task_num = sum(len(tasks_dict[i]) for i in range(step_num))
# print(f"all_task_num: {all_task_num}, step_num: {step_num}")

MD={}
COS_SIM_MATRIX = {}
for i in range(0, step_num):
    feat_path = f"{backbone}_class_feature_mean/{dataset}_{BB}-0.001_{task_name}_{i}.npy"
    if probing:
        embed_path = f"{backbone}_class_embedding_probing/{dataset}_{BB}-FixB-{lr}_{task_name}-probing_{i+1}.npy"
    else:
        embed_path = f"{backbone}_class_embedding/{dataset}_{BB}-0.001_{task_name}_{i}.npy"
    feature_matrix = np.load(feat_path)  # [all_task_num, 256]
    embedding_matrix = np.load(embed_path)  # [all_task_num, 256]
    # print(f"size of feature_matrix: {feature_matrix.shape}, size of embedding_matrix: {embedding_matrix.shape}")
    
    # 计算21x21的余弦相似度矩阵
    cos_sim = np.zeros((all_task_num, all_task_num))
    for m in range(all_task_num):
        for n in range(all_task_num):
            dot = np.dot(embedding_matrix[m], feature_matrix[n])
            norm_m = np.linalg.norm(embedding_matrix[m])
            norm_n = np.linalg.norm(feature_matrix[n])
            cos_sim[m][n] = dot / (norm_m * norm_n + 1e-8)
    COS_SIM_MATRIX[i] = cos_sim

for i in range(0, step_num): # 1
    MD[i] = []
    sim_mtx = []
    labels = list(tasks_dict[i])
    labels_old = [label for s in range(i) for label in tasks_dict[s]]
    start = len(labels_old)
    end = len(labels_old) + len(labels)
    for j in range(i, step_num): 
        sim_mtx.append(COS_SIM_MATRIX[j][start:end][:])
        MD[i].append(np.mean(np.abs(sim_mtx[j-i]-sim_mtx[0])))

plt.figure(figsize=(8, 6))

if step_num == 11:
    # 创建自定义颜色渐变（从#F2F958到#300892）
    # cmap_1 = LinearSegmentedColormap.from_list('custom_gradient', ['#F2F958', '#DA775F'])
    # cmap_2 = LinearSegmentedColormap.from_list('custom_gradient', ['#DA775F', '#300892'])
    # cmap_1 = LinearSegmentedColormap.from_list('custom_gradient', ['#FF9300', '#8100FF'])
    # cmap_2 = LinearSegmentedColormap.from_list('custom_gradient', ['#8100FF', '#00FDFF'])
    cmap_1 = LinearSegmentedColormap.from_list('custom_gradient', ['#00FDFF', '#8100FF'])
    cmap_2 = LinearSegmentedColormap.from_list('custom_gradient', ['#8100FF', '#FF9300'])
    tasks = sorted([task for task in MD if task != 0])  # 忽略MD[0]并排序
    colors_1 = [cmap_1(i/3) for i in range(4)]  # 生成10个渐变颜色
    colors_2 = [cmap_2((i+1)/6) for i in range(6)]
    colors = colors_1 + colors_2
elif step_num == 6:
    cmap_1 = LinearSegmentedColormap.from_list('custom_gradient', ['#00FDFF', '#8100FF'])
    cmap_2 = LinearSegmentedColormap.from_list('custom_gradient', ['#8100FF', '#FF9300'])
    tasks = sorted([task for task in MD if task != 0])  # 忽略MD[0]并排序
    colors_1 = [cmap_1(i/1) for i in range(2)]  # 生成5个渐变颜色
    colors_2 = [cmap_2((i+1)/3) for i in range(3)]
    colors = colors_1 + colors_2
else:
    raise ValueError("Unsupported step_num. Only 6 or 11 are supported.")


# 为每个任务绘制折线
for idx, task in enumerate(tasks):
    x = np.arange(task, task + len(MD[task]))
    y = MD[task]
    plt.plot(x, y, 
             marker='*',         # 五角星标记
             markersize=10,       # 标记尺寸
             linewidth=4,        # 折线加粗
             color=colors[idx],  # 渐变颜色
             label=f'Step {task}')

# 坐标轴设置
plt.xticks(np.arange(0, step_num), labels=np.arange(0, step_num) + 1)
plt.xlim(0, step_num)  # 横轴延长到step_num
if task_name == '100-5':
    plt.ylim(0, 0.4) 
elif task_name == '100-10':
    plt.ylim(0, 0.3)
elif task_name == '10-1':
    plt.ylim(0, 0.8)
elif task_name == '15-5s':
    plt.ylim(0, 0.5)

# 移除网格
plt.grid(False)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
ax.yaxis.grid(True, linestyle='--', alpha=0.3)  # Add horizontal grid lines


plt.xlabel("Steps Has Learned", fontsize=18, labelpad=10)
plt.ylabel("Moving Distance", fontsize=18, labelpad=10)
# plt.title("(b) Probing Classifier", fontsize=18, pad=20)
plt.title(f"(a) {flag} Classifier",
          fontsize=25,
          pad=15,
          y=-0.25,
          loc='center')

# 图例设置（左上角）
plt.legend(ncol=2, 
           loc='upper left',
           frameon=True,
           framealpha=0.9,
           edgecolor='#D8D8D8')


# 优化布局并保存
base_path = "/data/Code/CSS/Rethinking/plotfigs/Figure3"
os.makedirs(base_path, exist_ok=True)
plt.tight_layout()
plt.savefig(f"{base_path}/{backbone}_{task_name}_{flag}.svg", dpi=600, bbox_inches='tight')
plt.show()

# 参数配置字典
figsize = (24, 8) if step_num == 11 else (12, 8)
config = {
    "figsize": figsize,
    "cmap": "Blues_r",
    # "cmap": "viridis",
    "title": f"(b) {flag} Cosine Similarity",
    "dpi": 300,
    "colorbar": {
        "position": [0.92, 0.15, 0.02, 0.7],
        "label": "Cosine Similarity"
    },
    "suptitle": {
        "y": 0.03,
        "fontsize": 14
    }
}
# colors = ["white", "#E6F1FF", "#B0D4FF", "#6BB2FF", "#1E88E5"]
# custom_cmap = LinearSegmentedColormap.from_list("white_blue", colors)
# config["cmap"] = custom_cmap


# 使用配置参数绘图
if step_num == 11:
    fig, axes = plt.subplots(2, 6, figsize=config["figsize"])
elif step_num == 6:
    fig, axes = plt.subplots(2, 3, figsize=config["figsize"])
plt.subplots_adjust(wspace=0.3, hspace=0.3)

for i in range(0, step_num):
    # [数据加载和计算部分保持不变...]
    # matrix = COS_SIM_MATRIX[i][1:, 1:].T #排除背景类
    matrix = COS_SIM_MATRIX[i].T #不排除背景类
    
    # 绘制热力图
    ax = axes[i//6, i%6] if step_num == 11 else axes[i//3, i%3]
    im = ax.imshow(matrix, cmap=config["cmap"], vmin=0, vmax=1)
    # 统一设置坐标轴标签
    ax.set_xlabel("Class Embeddings", fontsize=10)
    ax.set_ylabel("Class Feature Mean", fontsize=10)

    # 微调刻度显示
    ax.tick_params(axis='both', length=0)  # 隐藏刻度线但保留坐标轴
    
    ax.set_title(f"Obs. Cls. after Step {i}", fontsize=12)

if step_num == 11:
    axes[1, 5].axis('off')  # 关闭第二行第六列的空位
else: 
    axes[1, 2].axis('off')
# 添加颜色条
cax = fig.add_axes(config["colorbar"]["position"])
fig.colorbar(im, cax=cax, label=config["colorbar"]["label"])

# 添加主标题
plt.suptitle(config["title"], 
             y=config["suptitle"]["y"],
             fontsize=config["suptitle"]["fontsize"],
             verticalalignment='bottom')

base_path = "/data/Code/CSS/Rethinking/plotfigs/Figure4"
os.makedirs(base_path, exist_ok=True)
plt.savefig(f"{base_path}/{backbone}_{task_name}_cosine_matrix_{flag}.svg", dpi=config["dpi"], bbox_inches='tight')