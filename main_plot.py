#!/usr/bin/env python
import h5py
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import importlib
import random
import os
import argparse
from utils.plot_utils import *
# from utils.plot_utils_diy import *
import torch
torch.manual_seed(0)

if __name__ == "__main__":                                         #LOCALtra1,FedAvg1,FEDPROX1,FEDGEN1,FEDALIGN1,Fedhkd1,FEDPKD11
    parser = argparse.ArgumentParser()                                 #LOCALtra,FedAvg,FEDPROX,FEDGEN,FEDALIGN,Fedhkd,FEDPKD
    parser.add_argument("--dataset", type=str, default="EMnist-alpha10-ratio1")  #EMnist-alpha0.1-ratio1,celeba-user20-agg5
    parser.add_argument("--algorithms", type=str, default="FedPKD,FedAvg+,LOCALtra", help='algorithm names separate by comma')  #LOCALtra,FedAvg,FedAvghkd                       pFed_DataFreeKD,FedGen,FedEnsemble,FedProx,FedDistill,FedAvg LOCALtra
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--learning_rate", type=float, default=0.01, help='learning rate.')
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--num_glob_iters", type=int, default=200)
    parser.add_argument("--min_acc", type=float, default=-1.0)
    parser.add_argument("--num_users", type=int, default=10, help='number of active users per epoch.')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gen_batch_size", type=int, default=32)
    parser.add_argument("--plot_legend", type=int, default=1, help='plot legend if set to 1, omitted otherwise.')
    parser.add_argument("--times", type=int, default=3, help='number of random seeds, starting from 1.')
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    args = parser.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(',')]
    title = 'epoch{}'.format(args.local_epochs)
    plot_results(args, algorithms)
