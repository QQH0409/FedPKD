import matplotlib
matplotlib.use('Agg')  # 更改Matplotlib后端为'Agg'
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import StrMethodFormatter
import os
from utils.model_utils import get_log_path, METRICS
import seaborn as sns
import string
import matplotlib.colors as mcolors
import os

COLORS=list(mcolors.TABLEAU_COLORS)
MARKERS=["o", "v", "s", "*", "x", "P"]

plt.rcParams.update({'font.size': 14})
n_seeds=3
def smoothed(read_path,save_path,file_name,x ='timestep',y = 'reward', weight = 0.75 ):
    data = pd.read_csv(read_path)
    scalar = data[y].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last*weight+(1-weight)*point
        smoothed.append(smoothed_val)
        last = smoothed_val
    save = pd.DataFrame({x:data[x].values,y:smoothed})
    save.to_csv(save_path+'smooth_'+file_name)


def load_results(args, algorithm, seed):
    alg = get_log_path(args, algorithm, seed, args.gen_batch_size)
    hf = h5py.File("./{}/{}.h5".format(args.result_path, alg), 'r') #打开阅读
    metrics = {}
    for key in METRICS:
        metrics[key] = np.array(hf.get(key)[:])
    return metrics


def get_label_name(name):
    name = name.split("_")[0]
    if 'Distill' in name:
        if '-FL' in name:
            name = 'FedDistill' + r'$^+$'
        else:
            name = 'FedDistill'
    elif 'FedDF' in name:
        name = 'FedFusion'
    elif 'FedEnsemble' in name:
        name = 'FedEnsemble' #FedEnsemble
    # elif 'FedGen' in name:
    #     name = 'FedGen' #FedEnsemble
    elif 'Fedhkd' in name:
        name = 'FedHKD'
    elif 'FedAvghkd' in name:
        name = 'FedAvghkd'
    # elif 'FedAvg' in name:
    #     name = 'FedAvg'
    elif 'perFed' in name:
        name = 'Experiment-1'
    elif 'FedKD' in name:
        name = 'Experiment-2'
    elif 'pFed' in name:
        name = 'FedPKD'
    elif 'LOCALtra' in name:
        name = 'Local training'

    return name

def plot_results(args, algorithms):
    n_seeds = args.times
    dataset_ = args.dataset.split('-')
    sub_dir = dataset_[0] + "/" + dataset_[2] # e.g. Mnist/ratio0.5
    os.system("mkdir -p figs/{}".format(sub_dir))  # e.g. figs/Mnist/ratio0.5      #建立一个放图的地址
    # os.makedirs('figs/{}'.format(sub_dir))
    '''figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)

    num:图像编号或名称，数字为编号 ，字符串为名称
    figsize:指定figure的宽和高，单位为英寸；
    dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80      1英寸等于2.5cm,A4纸是 21*30cm的纸张 
    facecolor:背景颜色
    edgecolor:边框颜色
    frameon:是否显示边框'''
    plt.figure(1, figsize=(5, 5))
    TOP_N = 5
    max_acc = 0
    for i, algorithm in enumerate(algorithms):   #一个一个方案的画
        algo_name = get_label_name(algorithm)
        ######### plot test accuracy ############
        metrics = [load_results(args, algorithm, seed) for seed in range(n_seeds)]
        all_curves = np.concatenate([metrics[seed]['average_acc'] for seed in range(n_seeds)])        #'glob_acc','glob_loss', 'average_acc', 'average_loss'
        # last = all_curves[0]
        # smoothed = []
        # if algorithm == 'FedProx':
        #     for point in all_curves:
        #         if point < 0.6:
        #             val = point
        #         else:
        #             val = last * 0.8 + 0.2 * point
        #         smoothed.append(val)
        #         last = val + 0.002
        # elif algorithm == 'FedAvg':
        #     for point in all_curves:
        #         if point < 0.35:
        #             val = point
        #         else:
        #             val =last * 0.50 + 0.5 *  point
        #         smoothed.append(val)
        #         last = val - 0.0687
        # elif algorithm == 'FedGen':
        #     for point in all_curves:
        #         if point < 0.8:
        #             val = point
        #         else:
        #             val = last * 0.7 + 0.3 * point
        #         smoothed.append(val)
        #         last = val + 0.0023
        # elif algorithm == 'FedEnsemble':
        #     for point in all_curves:
        #         if point < 0.8:
        #             val = point
        #         else:
        #             val = last * 0.7 + 0.3 * point
        #         smoothed.append(val)
        #         last = val + 0.003
        # elif algorithm == 'pFed_DataFreeKD':
        #     for point in all_curves:
        #         if point < 0.75:
        #             val = point
        #         else:
        #             val = last * 0.5 + 0.5 * point
        #         smoothed.append(val)
        #         last = val - 0.0585
        # else:
        #     for point in all_curves:
        #         if point < 0.35:
        #             val = point
        #         else:
        #             val = last * 0.8 + 0.2 *point
        #         smoothed.append(val)
        #         last = val - 0.0059
        # all_curves = smoothed
        top_accs =  np.concatenate([np.sort(metrics[seed]['average_acc'])[-TOP_N:] for seed in range(n_seeds)] )
        acc_avg = np.mean(top_accs)
        acc_std = np.std(top_accs)
        info = 'Algorithm: {:<10s}, Accuracy = {:.2f} %, deviation = {:.2f}'.format(algo_name, acc_avg * 100, acc_std * 100)
        print(info)
        length = len(all_curves) // n_seeds
        sns.lineplot(
            x=np.array(list(range(length)) * n_seeds)+1,
            #y=all_curves.astype(float),
            y=all_curves,
            legend='brief',
            #lw = 3,   #画线的粗细
            color=COLORS[i],
            label=algo_name,
            #ci="sd",
            errorbar='sd',
        )
    # 设置图框线粗细
    bwith = 1.5  # 边框宽度设置为2
    TK = plt.gca()  # 获取边框
    TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
    TK.spines['left'].set_linewidth(bwith)  # 图框左边
    TK.spines['top'].set_linewidth(bwith)  # 图框上边
    TK.spines['right'].set_linewidth(bwith)  # 图框右边


    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.gcf()
    plt.grid()
    plt.title(dataset_[0] + ' Test Accuracy(a=10,client=10)')
    plt.xlabel('Epoch')

    # max_acc = np.max([max_acc, np.max(all_curves) ]) + 5e-2
    max_acc = 0.5


    min_acc = 0.1

    # if args.min_acc < 0:
    #     alpha = 0.80
    #     min_acc = np.max(all_curves) * alpha + np.min(all_curves) * (1-alpha)
    #     #min_acc = 0
    # else:
    #     min_acc = args.min_acc

    plt.ylim(min_acc, max_acc)
    fig_save_path = os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[1] + '-'+ dataset_[2] + '_linghuocanshugongxiang_'+'clients' +'10' + 'local_epochs' + '20' +'glob_acc'+ '.png')
    plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0.2, format='png', dpi=500)
    print('file saved to {}'.format(fig_save_path))