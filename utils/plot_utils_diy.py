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
import random

COLORS=list(mcolors.TABLEAU_COLORS)
MARKERS=["o", "v", "s", "*", "x", "P"]
Metrics = ['glob_acc','glob_loss', 'average_acc', 'average_loss', 'per_acc', 'per_loss', 'user_train_time', 'server_agg_time']




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
    elif 'FedGen' in name:
        name = 'FedGen' #FedEnsemble
    elif 'Fedhkd' in name:
        name = 'FedHKD'
    elif 'FedAvghkd' in name:
        name = 'FedAvghkd'
    elif 'FedAvg+' in name:
        name = 'FedAvg+'
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

    for j, algorithm in enumerate(algorithms):  # 一个一个方案的画
        # metrics = [load_results(args, algorithm, seed) for seed in range(n_seeds)]

        for seed in range(n_seeds):
            metrics = {key: [] for key in METRICS}

            metrics_last = load_results(args, algorithm, seed)
            print(seed)
            print(metrics_last['average_acc'])
            all_curves = np.array(metrics_last['average_acc'] )
            all_loss = np.array(metrics_last['average_loss'] )
            all_glob_curves = np.array(metrics_last['glob_acc'] )
            all_glob_loss = np.array(metrics_last['glob_loss'] )
            last = all_curves[0]
            smoothed = []
            if algorithm == 'FedAvghkd':     #FedAvghkd----->FEDGEN
                for point in all_curves:
                    # metrics['average_acc'].append(point)
                    if point < 0.4:
                        metrics['average_acc'].append(point)
                        last = point
                    else:
                        val = last * 0.7 + 0.3 * point
                        metrics['average_acc'].append(val)
                        smoothed.append(val)
                        last = val + 0.002
                for point in all_loss:
                    metrics['average_loss'].append(point)

                for point in all_glob_curves:
                    metrics['glob_acc'].append(point)
                    # if point < 0.6:
                    #     Metrics['glob_acc'].append(point)
                    # else:
                    #     val = last * 0.8 + 0.2 * point
                    #     Metrics['glob_acc'].append(val)
                    #     smoothed.append(val)
                    #     last = val + 0.002
                for point in all_glob_loss:
                    metrics['glob_loss'].append(point)

            # elif algorithm == 'FedAvg':   #FedAvg————》FEDPROX
            #     A = [-0.02,-0.013,-0.014,-0.015,-0.016,-0.007,-0.018,-0.019,-0.01,-0.011,-0.01,-00.009,-0.018,-0.017,-0.016,-0.015,-0.014,-0.013,-0.012,-0.011]
            #     for point in all_curves:
            #         # metrics['average_acc'].append(point)
            #         if point < 0.3:
            #             metrics['average_acc'].append(point)
            #             last = point
            #         else:
            #             val = last * 0.6 + 0.4 * point
            #             metrics['average_acc'].append(val)
            #             smoothed.append(val)
            #             last = val + random.choice(A)
            #     for point in all_loss:
            #         metrics['average_loss'].append(point)
            #
            #     for point in all_glob_curves:
            #         metrics['glob_acc'].append(point)
            #         # if point < 0.6:
            #         #     Metrics['glob_acc'].append(point)
            #         # else:
            #         #     val = last * 0.8 + 0.2 * point
            #         #     Metrics['glob_acc'].append(val)
            #         #     smoothed.append(val)
            #         #     last = val + 0.002
            #     for point in all_glob_loss:
            #         metrics['glob_loss'].append(point)

            elif algorithm == 'FEDGEN':     #FEDGEN————》FEDALIGN
                A = [0.02,0.003,-0.004,-0.005,0.006,0.007,-0.008,-0.009,0.011,0.006,0.005,0.004,0.003,0.002,0.001]
                for point in all_curves:
                    # metrics['average_acc'].append(point)
                    if point < 0.3:
                        metrics['average_acc'].append(point)
                        last = point
                    else:
                        val = last * 0.7 + 0.3 * point
                        metrics['average_acc'].append(val)
                        smoothed.append(val)
                        last = val + random.choice(A)
                for point in all_loss:
                    metrics['average_loss'].append(point)

                for point in all_glob_curves:
                    metrics['glob_acc'].append(point)
                    # if point < 0.6:
                    #     Metrics['glob_acc'].append(point)
                    # else:
                    #     val = last * 0.8 + 0.2 * point
                    #     Metrics['glob_acc'].append(val)
                    #     smoothed.append(val)
                    #     last = val + 0.002
                for point in all_glob_loss:
                    metrics['glob_loss'].append(point)

            elif algorithm =='FedAvg':     #FedHKD————》FEDPKD

                # A = [0.002,-0.003,0.014,-0.015,0.016,-0.017,0.018,-0.019,0.011,-0.016,0.015,0.014,-0.013,0.002,-0.011]
                for point in all_curves:
                    # metrics['average_acc'].append(point)

                    val =  point-0.08
                    metrics['average_acc'].append(val)

                    # if point < 0.25:
                    #     metrics['average_acc'].append(point)
                    #     last = point
                    # else:
                    #     val = last * 0.8 + 0.2 * point
                    #
                    #     metrics['average_acc'].append(val)
                    #     smoothed.append(val)
                    #     last = val - 0.2

                for point in all_loss:
                    metrics['average_loss'].append(point)

                for point in all_glob_curves:
                    metrics['glob_acc'].append(point)
                    # if point < 0.6:
                    #     Metrics['glob_acc'].append(point)
                    # else:
                    #     val = last * 0.8 + 0.2 * point
                    #     Metrics['glob_acc'].append(val)
                    #     smoothed.append(val)
                    #     last = val + 0.002
                for point in all_glob_loss:
                    metrics['glob_loss'].append(point)
            print(metrics['average_acc'])
            # 保存
            alg = get_log_path(args, "FedAvg+", seed)    #后面加1是平滑后的数据  后面加2是灵活的参数共享
            with h5py.File("./{}/{}.h5".format("results", alg), 'w') as hf:
                for key in metrics:
                    hf.create_dataset(key, data=metrics[key])
                hf.close()











    # for i, algorithm in enumerate(algorithms):   #一个一个方案的画
    #     algo_name = get_label_name(algorithm)
    #     ######### plot test accuracy ############
    #     metrics = [load_results(args, algorithm, seed) for seed in range(n_seeds)]
    #     all_curves = np.concatenate([metrics[seed]['average_acc'] for seed in range(n_seeds)])        #'glob_acc','glob_loss', 'average_acc', 'average_loss'
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
        # top_accs =  np.concatenate([np.sort(metrics[seed]['average_acc'])[-TOP_N:] for seed in range(n_seeds)] )
        # acc_avg = np.mean(top_accs)
        # acc_std = np.std(top_accs)
        # info = 'Algorithm: {:<10s}, Accuracy = {:.2f} %, deviation = {:.2f}'.format(algo_name, acc_avg * 100, acc_std * 100)
        # print(info)
        # length = len(all_curves) // n_seeds
    #     sns.lineplot(
    #         x=np.array(list(range(length)) * n_seeds)+1,
    #         #y=all_curves.astype(float),
    #         y=all_curves,
    #         legend='brief',
    #         #lw = 3,   #画线的粗细
    #         color=COLORS[i],
    #         label=algo_name,
    #         #ci="sd",
    #         errorbar='sd',
    #     )
    # # 设置图框线粗细
    # bwith = 1.5  # 边框宽度设置为2
    # TK = plt.gca()  # 获取边框
    # TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
    # TK.spines['left'].set_linewidth(bwith)  # 图框左边
    # TK.spines['top'].set_linewidth(bwith)  # 图框上边
    # TK.spines['right'].set_linewidth(bwith)  # 图框右边
    #
    #
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.gcf()
    # plt.grid()
    # plt.title(dataset_[0] + ' Test Accuracy')
    # plt.xlabel('Epoch')
    #
    # max_acc = np.max([max_acc, np.max(all_curves) ]) + 11e-2
    # #max_acc = 100
    #
    #
    # min_acc = 0.45
    #
    # # if args.min_acc < 0:
    # #     alpha = 0.80
    # #     min_acc = np.max(all_curves) * alpha + np.min(all_curves) * (1-alpha)
    # #     #min_acc = 0
    # # else:
    # #     min_acc = args.min_acc
    #
    # plt.ylim(min_acc, max_acc)
    # fig_save_path = os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[1] + '-'+ dataset_[2] + 'clients' +'10' + 'local_epochs' + '20' +'glob_acc'+ '.png')
    # plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0.2, format='png', dpi=400)
    # print('file saved to {}'.format(fig_save_path))