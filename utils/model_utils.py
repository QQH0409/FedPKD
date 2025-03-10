import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import random
import numpy as np
from FLAlgorithms.trainmodel.models import Net,CNNFemnist
from torch.utils.data import DataLoader
from FLAlgorithms.trainmodel.generator import Generator
from utils.model_config import *
METRICS = ['glob_acc','glob_loss', 'average_acc', 'average_loss', 'per_acc', 'per_loss', 'user_train_time', 'server_agg_time']


def get_data_dir(dataset):
    if 'EMnist' in dataset:
        #EMnist-alpha0.1-ratio0.1-0-letters
        dataset_=dataset.replace('alpha', '').replace('ratio', '').split('-')
        alpha, ratio =dataset_[1], dataset_[2]
        types = 'letters'
        # path_prefix = os.path.join('data', 'EMnist', f'u20-{types}-alpha{alpha}-ratio{ratio}')
        path_prefix = os.path.join('data', 'EMnist', f'u100-{types}-alpha{alpha}-ratio{ratio}')
        train_data_dir=os.path.join(path_prefix, 'train')
        test_data_dir=os.path.join(path_prefix, 'test')
        proxy_data_dir = 'data/proxy_data/emnist-n10/'

    elif 'Mnist' in dataset:
        dataset_=dataset.replace('alpha', '').replace('ratio', '').split('-')
        alpha, ratio=dataset_[1], dataset_[2]
        # path_prefix=os.path.join('data', 'Mnist', 'u20c10-alpha{}-ratio{}'.format(alpha, ratio))
        path_prefix=os.path.join('data', 'Mnist', 'u100c10-alpha{}-ratio{}'.format(alpha, ratio))
        train_data_dir=os.path.join(path_prefix, 'train')
        test_data_dir=os.path.join(path_prefix, 'test')
        proxy_data_dir = 'data/proxy_data/mnist-n10/'

    elif 'celeb' in dataset.lower():
        dataset_ = dataset.lower().replace('user', '').replace('agg','').split('-')
        user, agg_user = dataset_[1], dataset_[2]
        path_prefix = os.path.join('data', 'CelebA', 'user{}-agg{}'.format(user,agg_user))
        train_data_dir=os.path.join(path_prefix, 'train')
        test_data_dir=os.path.join(path_prefix, 'test')
        proxy_data_dir=os.path.join('/user500/', 'proxy')

    else:
        raise ValueError("Dataset not recognized.")
    return train_data_dir, test_data_dir, proxy_data_dir


def read_data(dataset):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_data_dir, test_data_dir, proxy_data_dir = get_data_dir(dataset)
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    proxy_data = {}



    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') or f.endswith(".pt")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        if file_path.endswith("json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        elif file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))

        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') or f.endswith(".pt")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        if file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))
        test_data.update(cdata['user_data'])


    if proxy_data_dir and os.path.exists(proxy_data_dir):
        proxy_files=os.listdir(proxy_data_dir)
        proxy_files=[f for f in proxy_files if f.endswith('.json') or f.endswith(".pt")]
        for f in proxy_files:
            file_path=os.path.join(proxy_data_dir, f)
            if file_path.endswith(".pt"):
                with open(file_path, 'rb') as inf:
                    cdata=torch.load(inf)
            elif file_path.endswith(".json"):
                with open(file_path, 'r') as inf:
                    cdata=json.load(inf)
            else:
                raise TypeError("Data format not recognized: {}".format(file_path))
            proxy_data.update(cdata['user_data'])

    return clients, groups, train_data, test_data, proxy_data

def read_global_test_data(dataset):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_data_dir, test_data_dir, proxy_data_dir = get_data_dir(dataset)
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    proxy_data = {}
    total_test_samples = 0


    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') or f.endswith(".pt")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        if file_path.endswith("json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        elif file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))

        # 更新总样本数（假设每个用户的'x'键包含一个列表）
        total_test_samples += len(cdata['user_data'][list(cdata['user_data'].keys())[0]]['x'])   # 假设所有用户有相同数量的样本，否则需要更复杂的逻辑

    # 随机选择一半样本的索引
    selected_indices = random.sample(range(total_test_samples), total_test_samples // 2)

    # 初始化global_data字典和跟踪每个用户的样本数
    global_data = {}
    user_sample_counts = {user_id: 0 for user_id in test_data}

    # 遍历test_data字典，根据索引选择样本
    for user_id, user_data in test_data.items():
        samples = user_data['x']
        labels = user_data['y'] if 'y' in user_data else None
        for sample, label in zip(samples, labels if labels is not None else [None] * len(samples)):
            # 增加当前用户的样本计数
            user_sample_counts[user_id] += 1
            # 检查当前样本的计数是否在我们选择的索引中
            if user_sample_counts[user_id] - 1 in selected_indices:  # 注意减1，因为selected_indices是基于0的索引
                if user_id not in global_data:
                    global_data[user_id] = {'x': [], 'y': []}
                global_data[user_id]['x'].append(sample)
                if label is not None:
                    global_data[user_id]['y'].append(label)

    return global_data




def read_proxy_data(proxy_data, dataset, batch_size):
    X, y=proxy_data['x'], proxy_data['y']
    X, y = convert_data(X, y, dataset=dataset)
    dataset = [(x, y) for x, y in zip(X, y)]
    proxyloader = DataLoader(dataset, batch_size, shuffle=True)
    iter_proxyloader = iter(proxyloader)
    return proxyloader, iter_proxyloader


def aggregate_data_(clients, dataset, dataset_name, batch_size):
    combined = []
    unique_labels = []
    for i in range(len(dataset)):
        id = clients[i]
        user_data = dataset[id]
        X, y = convert_data(user_data['x'], user_data['y'], dataset=dataset_name)
        combined += [(x, y) for x, y in zip(X, y)]
        unique_y=torch.unique(y)
        unique_y = unique_y.detach().numpy()
        unique_labels += list(unique_y)

    data_loader=DataLoader(combined, batch_size, shuffle=True)
    iter_loader=iter(data_loader)
    return data_loader, iter_loader, unique_labels


def aggregate_user_test_data(data, dataset_name, batch_size):
    clients, loaded_data=data[0], data[3]
    data_loader, _, unique_labels=aggregate_data_(clients, loaded_data, dataset_name, batch_size)
    return data_loader, np.unique(unique_labels)


def aggregate_user_data(data, dataset_name, batch_size):
    # data contains: clients, groups, train_data, test_data, proxy_data
    clients, loaded_data = data[0], data[2]
    data_loader, data_iter, unique_labels = aggregate_data_(clients, loaded_data, dataset_name, batch_size)
    return data_loader, data_iter, np.unique(unique_labels)


def convert_data(X, y, dataset=''):
    if not isinstance(X, torch.Tensor):
        if 'celeb' in dataset.lower():
            X=torch.Tensor(X).type(torch.float32).permute(0, 3, 1, 2)
            y=torch.Tensor(y).type(torch.int64)

        else:
            X=torch.Tensor(X).type(torch.float32)
            y=torch.Tensor(y).type(torch.int64)
    return X, y


def read_user_data(index, data, dataset='', count_labels=False):       #index参预训练的用户的索引
    #data contains: clients, groups, train_data, test_data, proxy_data(optional)
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train = convert_data(train_data['x'], train_data['y'], dataset=dataset)
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    X_test, y_test = convert_data(test_data['x'], test_data['y'], dataset=dataset)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    if count_labels:
        label_info = {}
        unique_y, counts=torch.unique(y_train, return_counts=True)
        unique_y=unique_y.detach().numpy()
        counts=counts.detach().numpy()
        label_info['labels']=unique_y
        label_info['counts']=counts
        return id, train_data, test_data, label_info
    return id, train_data, test_data


def get_dataset_name(dataset):
    dataset=dataset.lower()
    passed_dataset=dataset.lower()
    if 'celeb' in dataset:
        passed_dataset='celeb'
    elif 'emnist' in dataset:
        passed_dataset='emnist'
    elif 'mnist' in dataset:
        passed_dataset='mnist'
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))
    return passed_dataset

# 生成器模型网络架构定义：
def create_generative_model(dataset, algorithm='', model='cnn', embedding=False): # 原来是embedding=False
    passed_dataset=get_dataset_name(dataset)
    assert any([alg in algorithm for alg in ['Fed', 'Fed']])     #原来是assert any([alg in algorithm for alg in ['FedGen', 'FedGen']])
    if 'Fed' in algorithm:    #原来是if 'FedGen' in algorithm:
        # temporary roundabout to figure out the sensitivity of the generator network & sampling size
        if 'cnn' in algorithm:
            gen_model = algorithm.split('-')[1]
            passed_dataset+='-' + gen_model
        elif '-pFed' in algorithm: # we use more lightweight network for sensitivity analysis 原来是
            passed_dataset += '-cnn1'
    return Generator(passed_dataset, model=model, embedding=embedding, latent_layer_idx=-1)


def create_model(model, dataset, algorithm):
    passed_dataset = get_dataset_name(dataset)
    model= Net(passed_dataset, model), model
    return model


def polyak_move(params, target_params, ratio=0.1):
    for param, target_param in zip(params, target_params):
        param.data=param.data - ratio * (param.clone().detach().data - target_param.clone().detach().data)

def meta_move(params, target_params, ratio):
    for param, target_param in zip(params, target_params):
        target_param.data = param.clone().data + ratio * (target_param.clone().data - param.clone().data)

def moreau_loss(params, reg_params):
    # return 1/T \sum_i^T |param_i - reg_param_i|^2
    losses = []
    for param, reg_param in zip(params, reg_params):
        losses.append( torch.mean(torch.square(param - reg_param.clone().detach())) )
    loss = torch.mean(torch.stack(losses))
    return loss

def l2_loss(params):
    losses = []
    for param in params:
        losses.append( torch.mean(torch.square(param)))
    loss = torch.mean(torch.stack(losses))
    return loss

def update_fast_params(fast_weights, grads, lr, allow_unused=False):
    """
    Update fast_weights by applying grads.
    :param fast_weights: list of parameters.
    :param grads: list of gradients
    :param lr:
    :return: updated fast_weights .
    """
    for grad, fast_weight in zip(grads, fast_weights):
        if allow_unused and grad is None: continue
        grad=torch.clamp(grad, -10, 10)
        fast_weight.data = fast_weight.data.clone() - lr * grad
    return fast_weights


def init_named_params(model, keywords=['encode']):
    named_params={}
    #named_params_list = []
    for name, params in model.named_layers.items():
        if any([key in name for key in keywords]):
            named_params[name]=[param.clone().detach().requires_grad_(True) for param in params]
            #named_params_list += named_params[name]
    return named_params#, named_params_list



def get_log_path(args, algorithm, seed, gen_batch_size=32):
    alg=args.dataset + "_" + algorithm
    alg+="_" + str(args.learning_rate) + "_" + str(args.num_users)
    alg+="u" + "_" + str(args.batch_size) + "b" + "_" + str(args.local_epochs)
    alg=alg + "_" + str(seed)
    # 原来是 if 'FedGen' in algorithm: # to accompany experiments for author rebuttal
    alg += "_embed" + str(args.embedding)
    if int(gen_batch_size) != int(args.batch_size):
        alg += "_gb" + str(gen_batch_size)
    return alg