from tqdm import trange
import numpy as np
import random
import json
import os
import argparse
from torchvision.datasets import MNIST
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
from sympy.physics.control.control_plots import plt
#import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

def rearrange_data_by_class(data, targets, n_class):
    new_data = []
    for i in trange(n_class):
        idx = targets == i
        new_data.append(data[idx])
    return new_data

def get_dataset(mode='train'):
    transform = transforms.Compose(
       [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    dataset = MNIST(root='./data', train=True if mode=='train' else False, download=True, transform=transform)
    n_sample = len(dataset.data)
    SRC_N_CLASS = len(dataset.classes)
    # full batch
    trainloader = DataLoader(dataset, batch_size=n_sample, shuffle=False)

    print("Loading data from storage ...")
    for _, xy in enumerate(trainloader, 0):
        dataset.data, dataset.targets = xy

    print("Rearrange data by class...")
    data_by_class = rearrange_data_by_class(
        dataset.data.cpu().detach().numpy(),
        dataset.targets.cpu().detach().numpy(),
        SRC_N_CLASS
    )
    print(f"{mode.upper()} SET:\n  Total #samples: {n_sample}. sample shape: {dataset.data[0].shape}")
    print("  #samples per class:\n", [len(v) for v in data_by_class])

    return data_by_class, n_sample, SRC_N_CLASS

def sample_class(SRC_N_CLASS, NUM_LABELS, user_id, label_random=False):
    assert NUM_LABELS <= SRC_N_CLASS
    if label_random:
        source_classes = [n for n in range(SRC_N_CLASS)]
        random.shuffle(source_classes)
        return source_classes[:NUM_LABELS]
    else:
        return [(user_id + j) % SRC_N_CLASS for j in range(NUM_LABELS)]

def devide_train_data(data, n_sample, SRC_CLASSES, NUM_USERS, min_sample, alpha=0.5, sampling_ratio=0.5):
    min_sample = 10#len(SRC_CLASSES) * min_sample
    min_size = 0 # track minimal samples per user
    ###### Determine Sampling #######
    while min_size < min_sample:
        print("Try to find valid data separation")
        idx_batch=[{} for _ in range(NUM_USERS)]
        samples_per_user = [0 for _ in range(NUM_USERS)]
        max_samples_per_user = sampling_ratio * n_sample / NUM_USERS
        for l in SRC_CLASSES:
            # get indices for all that label
            idx_l = [i for i in range(len(data[l]))]
            np.random.shuffle(idx_l)
            if sampling_ratio < 1:
                samples_for_l = int( min(max_samples_per_user, int(sampling_ratio * len(data[l]))) )
                idx_l = idx_l[:samples_for_l]
                print(l, len(data[l]), len(idx_l))
            # dirichlet sampling from this label
            proportions=np.random.dirichlet(np.repeat(alpha, NUM_USERS))
            # re-balance proportions
            proportions=np.array([p * (n_per_user < max_samples_per_user) for p, n_per_user in zip(proportions, samples_per_user)])
            proportions=proportions / proportions.sum()
            proportions=(np.cumsum(proportions) * len(idx_l)).astype(int)[:-1]
            # participate data of that label
            for u, new_idx in enumerate(np.split(idx_l, proportions)):
                # add new idex to the user
                idx_batch[u][l] = new_idx.tolist()
                samples_per_user[u] += len(idx_batch[u][l])
        min_size=max(samples_per_user)    #原来是min_size=min(samples_per_user)

    ###### CREATE USER DATA SPLIT #######
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    Labels=[set() for _ in range(NUM_USERS)]
    print("processing users...")
    for u, user_idx_batch in enumerate(idx_batch):
        for l, indices in user_idx_batch.items():
            if len(indices) == 0: continue
            X[u] += data[l][indices].tolist()
            y[u] += (l * np.ones(len(indices))).tolist()
            Labels[u].add(l)

    return X, y, Labels, idx_batch, samples_per_user

def divide_test_data(NUM_USERS, SRC_CLASSES, test_data, Labels, unknown_test):
    # Create TEST data for each user.
    test_X = [[] for _ in range(NUM_USERS)]
    test_y = [[] for _ in range(NUM_USERS)]
    idx = {l: 0 for l in SRC_CLASSES}
    for user in trange(NUM_USERS):
        if unknown_test: # use all available labels
            user_sampled_labels = SRC_CLASSES
        else:
            user_sampled_labels =  list(Labels[user])
        for l in user_sampled_labels:
            num_samples = int(len(test_data[l]) / NUM_USERS )
            assert num_samples + idx[l] <= len(test_data[l])
            test_X[user] += test_data[l][idx[l]:idx[l] + num_samples].tolist()
            test_y[user] += (l * np.ones(num_samples)).tolist()
            assert len(test_X[user]) == len(test_y[user]), f"{len(test_X[user])} == {len(test_y[user])}"
            idx[l] += num_samples
    return test_X, test_y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", "-f", type=str, default="pt", help="Format of saving: pt (torch.save), json", choices=["pt", "json"])
    parser.add_argument("--n_class", type=int, default=10, help="number of classification labels")
    parser.add_argument("--min_sample", type=int, default=10, help="Min number of samples per user.")
    parser.add_argument("--sampling_ratio", type=float, default=1, help="Ratio for sampling training samples.")
    parser.add_argument("--unknown_test", type=int, default=0, help="Whether allow test label unseen for each user.")
    parser.add_argument("--alpha", type=float, default=100, help="alpha in Dirichelt distribution (smaller means larger heterogeneity)")
    parser.add_argument("--n_user", type=int, default=100,
                        help="number of local clients, should be muitiple of 10.")
    args = parser.parse_args()
    print()
    print("Number of users: {}".format(args.n_user))
    print("Number of classes: {}".format(args.n_class))
    print("Min # of samples per uesr: {}".format(args.min_sample))
    print("Alpha for Dirichlet Distribution: {}".format(args.alpha))
    print("Ratio for Sampling Training Data: {}".format(args.sampling_ratio))
    NUM_USERS = args.n_user

    # Setup directory for train/test data
    path_prefix = f'u{args.n_user}c{args.n_class}-alpha{args.alpha}-ratio{args.sampling_ratio}'

    def process_user_data(mode, data, n_sample, SRC_CLASSES, Labels=None, unknown_test=0):
        if mode == 'train':
            X, y, Labels, idx_batch, samples_per_user  = devide_train_data(
                data, n_sample, SRC_CLASSES, NUM_USERS, args.min_sample, args.alpha, args.sampling_ratio)
        if mode == 'test':
            assert Labels != None or unknown_test
            X, y = divide_test_data(NUM_USERS, SRC_CLASSES, data, Labels, unknown_test)
        dataset={'users': [], 'user_data': {}, 'num_samples': []}
        for i in range(NUM_USERS):
            uname='f_{0:05d}'.format(i)
            dataset['users'].append(uname)
            dataset['user_data'][uname]={
                'x': torch.tensor(X[i], dtype=torch.float32),
                'y': torch.tensor(y[i], dtype=torch.float32)}     #原来是dtype=torch.int64
            dataset['num_samples'].append(len(X[i]))

        print("{} #sample by user:".format(mode.upper()), dataset['num_samples'])

        data_path=f'./{path_prefix}/{mode}'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        data_path=os.path.join(data_path, "{}.".format(mode) + args.format)
        if args.format == "json":
            raise NotImplementedError(
                "json is not supported because the train_data/test_data uses the tensor instead of list and tensor cannot be saved into json.")
            with open(data_path, 'w') as outfile:
                print(f"Dumping train data => {data_path}")
                json.dump(dataset, outfile)
        elif args.format == "pt":
            with open(data_path, 'wb') as outfile:
                print(f"Dumping train data => {data_path}")
                torch.save(dataset, outfile)
        if mode == 'train':
            plt_matrix = np.zeros((10, 100), int)
            for u in range(NUM_USERS):
                print("{} samples in total".format(samples_per_user[u]))
                train_info = ''
                # train_idx_batch, train_samples_per_user
                n_samples_for_u = 0
                for l in sorted(list(Labels[u])):
                    n_samples_for_l = len(idx_batch[u][l])
                    n_samples_for_u += n_samples_for_l
                    train_info += "c={},n={}| ".format(l, n_samples_for_l)
                    plt_matrix[l][u] = n_samples_for_l
                print(train_info)
                print("{} Labels/ {} Number of training samples for user [{}]:".format(len(Labels[u]), n_samples_for_u, u))

            # 画图 保存
            # 定义热图的横纵坐标
            yLabel = ['0', '1', '2', '3', '4', '5', '6','7', '8', '9']
            xLabel = ['0', '1', '2', '3', '4', '5','6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                          '18', '19','20', '21', '22', '23', '24', '25','26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
                          '38', '39','40', '41', '42', '43', '44', '45','46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57',
                          '58', '59','60', '61', '62', '63', '64', '65','66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77',
                          '78', '79','80', '81', '82', '83', '84', '85','86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97',
                          '98', '99']

            # 准备数据阶段
            data = plt_matrix
            # 创建一个新的图形，设置宽度为12英寸，高度为4英寸
            plt.figure(figsize=(50, 10))
            # # 作图阶段
            fig, ax = plt.subplots()
            # 定义横纵坐标的刻度
            ax.set_yticks(range(len(yLabel)))
            ax.set_yticklabels(yLabel)
            ax.set_xticks(range(len(xLabel)))
            ax.set_xticklabels(xLabel)
            # 修改y轴刻度标签的大小和旋转
            ax.tick_params(axis='y', labelsize=3, rotation=0)  # 例如，将y轴标签大小设置为12，不旋转
            ax.tick_params(axis='x', labelsize=3, rotation=45)  # 将x轴标签大小设置为10，旋转45度

            # 设置刻度线的粗细
            # 注意：matplotlib的刻度线粗细通常通过tick_params的width参数来设置,长度length,与标签的距离pad
            ax.tick_params(axis='both', which='major', length=1,width=0.1,pad=2)  # 将主刻度线的粗细设置为0.5
            ax.tick_params(axis='both', which='minor', length=1,width=0.1,pad=2)  # 如果有次刻度线，也可以设置其粗细


            # 作图并选择热图的颜色填充风格，这里选择yLGn
            im = ax.imshow(data, cmap="bwr", alpha=0.5, interpolation='nearest', origin='lower', vmin=0, vmax=300)
            # 增加右侧的颜色刻度条
            plt.colorbar(im,fraction=0.01)
            # plt.colorbar(im.colorbar, fraction=0.025)

            # 填充数字
            for i in range(len(yLabel)):
                for j in range(len(xLabel)):
                    print('data[{},{}]:{}'.format(i, j, data[i, j]))
                    ax.text(j, i, data[i, j],
                            ha="center", va="center", color="black",size ='1.8')

            # 设置坐标轴线条的粗细（默认为1.0，可以调整）
            for spine in ax.spines.values():
                spine.set_linewidth(0.2)  # 设置坐标轴线条的粗细为0.5
            # 手动设置x轴的刻度位置
            #ax.set_xticks([1, 2.5, 4])  # 刻度线将出现在x=1, 2.5, 4的位置

            # 增加标题
            plt.title("Mnist alpha100", fontdict={'size': 16})
            plt.xlabel(r'user',rotation=0, fontdict={'size': 16})
            plt.ylabel(r'2', rotation=0, fontdict={'size': 16})
            # show
            fig.tight_layout()
            # plt.show()
            # im2=plt.matshow(plt_matrix, cmap=plt.get_cmap('Blues'), alpha=0.5, interpolation='nearest',origin ='lower',vmin = 0, vmax =2000)
            fig_save_path = os.path.join('figs' + '/' + 'client100-alpha100' + '.png')
            plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0, format='png', dpi=1500)
            print('file saved to {}'.format(fig_save_path))
            return Labels, idx_batch, samples_per_user


    print(f"Reading source dataset.")
    train_data, n_train_sample, SRC_N_CLASS = get_dataset(mode='train')
    test_data, n_test_sample, SRC_N_CLASS = get_dataset(mode='test')
    SRC_CLASSES=[l for l in range(SRC_N_CLASS)]
    random.shuffle(SRC_CLASSES)
    print("{} labels in total.".format(len(SRC_CLASSES)))
    Labels, idx_batch, samples_per_user = process_user_data('train', train_data, n_train_sample, SRC_CLASSES)
    process_user_data('test', test_data, n_test_sample, SRC_CLASSES, Labels=Labels, unknown_test=args.unknown_test)
    print("Finish Generating User samples")

if __name__ == "__main__":
    main()