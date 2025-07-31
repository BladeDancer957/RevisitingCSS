import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# 设置全局字体为Arial并加粗
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'
plt.style.use('seaborn-v0_8-whitegrid')

base_path = "/data/Code/CSS/Rethinking/plotfigs/Figure5"
os.makedirs(base_path, exist_ok=True)

# 数据预处理
data = {
    'Res101': {
        '10-1': {
            'Probing': [0.3592352867126465, 0.6486257314682007, 0.5462563037872314, 0.5533873438835144, 0.5386977195739746, 0.5254482626914978, 0.4724363684654236, 0.4420085549354553, 0.36280593276023865, 0.4292084872722626, 0.4147109091281891, 0.3904717266559601],
            'FT': [0, 0.8110582232475281, 0.23207120597362518, 0.13575239479541779, 0.11335394531488419, 0.07273325324058533, 0.0950850248336792, 0.05109154433012009, 0.047762688249349594, 0.04113372042775154, 0.04155988618731499, 0.036392420530319214]
        },
        '15-1': {
            'Probing': [0.3592408299446106, 0.7101538777351379, 0.6342671513557434, 0.6226193904876709, 0.6010395288467407, 0.6008548140525818, 0.5738885998725891] + [np.nan]*5,
            'FT': [0, 0.7965526580810547, 0.1254306137561798, 0.1727718710899353, 0.07058918476104736, 0.04362133890390396, 0.039738673716783524] + [np.nan]*5 
        },
        '100-5':{
            'Probing': [0.15726858377456665, 0.33425483107566833, 0.2768988311290741, 0.26632067561149597, 0.261809766292572, 0.25824233889579773, 0.25117555260658264, 0.25271913409233093, 0.2519342303276062, 0.2539956867694855, 0.24929435551166534, 0.24803540110588074],
            'FT': [0, 0.4248276650905609, 0.009312370792031288, 0.010626036673784256, 0.009061760269105434, 0.008173491805791855, 0.008705618791282177, 0.010107839480042458, 0.007059857249259949, 0.005621293559670448, 0.0035317069850862026, 0.003802982857450843]
        },
        '100-10':{
            'Probing': [0.1572698950767517, 0.3362421989440918, 0.26576343178749084, 0.25932177901268005, 0.24846230447292328, 0.2539132833480835, 0.25697842240333557] + [np.nan]*5,
            'FT': [0, 0.423485666513443, 0.012204837054014206, 0.019011376425623894, 0.01899774931371212, 0.011760863475501537, 0.006661824882030487] + [np.nan]*5
        }
    },
    'SwinB': {
        '10-1': {
            'Probing': [0.35312265157699585, 0.7394477128982544, 0.7238215804100037, 0.7216612100601196, 0.7222700715065002, 0.725713849067688, 0.7057158350944519, 0.6961222887039185, 0.7034844160079956, 0.7117236256599426, 0.7379032969474792, 0.7394741177558899],
            'FT': [0, 0.8194640874862671, 0.39618852734565735, 0.22892270982265472, 0.17982281744480133, 0.12733446061611176, 0.09964416921138763, 0.06649056822061539, 0.07456425577402115, 0.04968397691845894, 0.05450623109936714, 0.05617085099220276] 
        },
        '15-1': {
            'Probing': [0.35309329628944397, 0.7782799005508423, 0.7185043692588806, 0.737695574760437, 0.7417693138122559, 0.7533990740776062, 0.761132538318634] + [np.nan]*5, 
            'FT': [0, 0.8242158889770508, 0.09763620793819427, 0.20014314353466034, 0.12935703992843628, 0.11134012043476105, 0.06975807994604111] + [np.nan]*5 
        },
        '100-5':{
            'Probing': [0.1781625598669052, 0.363211452960968, 0.22210446000099182, 0.23359978199005127, 0.2208702713251114, 0.2214917540550232, 0.2349337339401245, 0.22949546575546265, 0.2394561767578125, 0.23291873931884766, 0.240507572889328, 0.2308584600687027], # 30 epoch
            'FT': [0, 0.428687185049057, 0.004818665329366922, 0.004733305890113115, 0.005886171478778124, 0.011507906019687653, 0.008974127471446991, 0.007161677815020084, 0.004816361237317324, 0.0040174913592636585, 0.0024949798826128244, 0.0006040236912667751]
        },
        '100-10':{
            'Probing': [0.17848405241966248, 0.365534245967865, 0.24280305206775665, 0.24041366577148438, 0.25541365146636963, 0.2609045207500458, 0.2609791159629822] + [np.nan]*5,
            'FT': [0, 0.43209579586982727, 0.008577137254178524, 0.015981346368789673, 0.014116267673671246, 0.005670947954058647, 0.0042145708575844765] + [np.nan]*5
        }

    }
}

dataset_list = ['voc', 'ade']
voc_task_list = ['10-1', '15-1']
ade_task_list = ['100-5', '100-10']
methods = ['Probing', 'FT']
# colors = {'Res101': '#1f77b4', 'SwinB': '#ff7f0e'}
# colors = {'Res101': '#D5818B', 'SwinB': '#9EB8E3'}
colors = {'Res101': '#9EB8E3', 'SwinB': '#D5818B'}

# 遍历每个任务和方法组合
for dataset in dataset_list:
    if dataset == 'voc':
        tasks = voc_task_list
    elif dataset == 'ade':
        tasks = ade_task_list
    for task in tasks:
        for method in methods:
            # 创建新figure
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            # 绘制两个backbone的曲线
            for backbone in ['Res101', 'SwinB']:
                # values = data[backbone][task][method]
                values = data[backbone][task][method][1:]  # 去掉第一个0值
                steps = np.arange(len(values)) + 1
                if method == 'FT':
                    if backbone == 'Res101':
                        marker = 'o'
                        markersize = 8
                    else:
                        marker = '*'
                        markersize = 11
                else:
                    if backbone == 'Res101':
                        marker = 'p'
                        markersize = 9
                    else:
                        marker = 'd'
                        markersize = 9
                ax.plot(steps, values, 
                        marker=marker,
                        color=colors[backbone],
                        linewidth=2,
                        markersize=markersize,
                        label=backbone)
            
            # 设置图表元素
            ax.set_title(f'{dataset} {task} {method}', fontsize=18, pad=10)
            ax.set_xlabel('Step', fontsize=18)
            ax.set_ylabel('mIoU', fontsize=18)
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            int_steps = np.arange(1, len([val for val in values if not np.isnan(val)]) + 1)  # 从1开始
            ax.set_xticks(int_steps)
            ax.set_xticklabels([str(int(x)) for x in int_steps])
            # 移除网格
            plt.grid(False)
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)  # Add horizontal grid lines
            # ax.grid(True, alpha=0.3)
            if dataset == 'voc':
                ax.set_ylim(0, 0.9)
            elif dataset == 'ade':
                ax.set_ylim(0, 0.45)
            ax.legend(loc='upper right', fontsize=16)
            ax.tick_params(axis='both', labelsize=16)
            
            # 保存并关闭当前figure
            plt.tight_layout()
            plt.savefig(f'{base_path}/{dataset}_{task}_{method}.svg', bbox_inches='tight')  # 保存为PDF
            plt.close()  # 关闭当前图表，释放内存

print("所有图表已保存完成！")