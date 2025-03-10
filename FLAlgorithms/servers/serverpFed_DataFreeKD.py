# （FedGen-main）2022/10/11 17:22
import scipy

from FLAlgorithms.users.userpFed_DataFreeKD import UserpFed_DataFreeKD
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import copy
import time
import re
from tqdm import *
MIN_SAMPLES_PER_LABEL=1

class pFed_DataFreeKD(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args.dataset)

        # data contains: clients, groups, train_data, test_data, proxy_data
        clients = data[0]
        total_users = len(clients)
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower()
        self.use_adam = 'adam' in self.algorithm.lower()

        self.early_stop = 20  # stop using generated samples after 20 local epochs
        self.student_model = copy.deepcopy(self.model)

        self.generative_model = create_generative_model(args.dataset, args.algorithm, self.model_name, args.embedding)

        if not args.train:            #args.train == 1
            print('number of generator parameteres: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = self.generative_model.latent_layer_idx
        print(self.latent_layer_idx)
        self.init_ensemble_configs()
        print("latent_layer_idx: {}".format(self.latent_layer_idx))
        print("label embedding {}".format(self.generative_model.embedding))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta, self.ensemble_eta))
        print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))
        self.init_loss_fn()
        self.train_data_loader, self.train_iter, self.available_labels = aggregate_user_data(data, args.dataset, self.ensemble_batch_size)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)

        #### creating users ####
        self.users = []
        for i in range(total_users):   #每一个参与训练的用户
            id, train_data, test_data, label_info =read_user_data(i, data, dataset=args.dataset, count_labels=True)
            self.total_train_samples+=len(train_data)
            self.total_test_samples += len(test_data)
            id, train, test=read_user_data(i, data, dataset=args.dataset)
            user=UserpFed_DataFreeKD(
                args, id, model, self.generative_model,
                train_data, test_data,
                self.available_labels, self.latent_layer_idx, label_info,
                use_adam=self.use_adam)
            self.users.append(user)
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating {} server.".format(args.algorithm))

    def train(self, args):
        # 放个性化生成模型的字典
        generator_userid_key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        generator_model_dict = dict([(k, self.generative_model.parameters()) for k in generator_userid_key])
        #### pretraining
        preds_userid_key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        preds_matrix = dict([(k, 0) for k in preds_userid_key])

        user_preds_userid_key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        user_preds_matrix = dict([(k, ([0,0,0],[0,0,0])) for k in user_preds_userid_key])
        user_train = []
        for glob_iter in range(self.num_glob_iters):       #200次循环迭代
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            # 第一轮先广播一个通用全局模型
            if glob_iter == 0:
               self.send_parameters(mode=self.mode)
            # 选择此轮训练的用户——在main.py中
            self.selected_users, self.user_idxs=self.select_users(glob_iter, self.num_users, return_idx=True)
            # 原来是——广播完模型以后 对所有用户得到的模型计算精度损失
            # self.evaluate()
            chosen_verbose_user = np.random.randint(0, len(self.users))  #np.random.randint返回一个随机用户
            self.timestamp = time.time() # log user-training start time
            for user_id, user in zip(self.user_idxs, self.selected_users): # allow selected users to train
                if user_id in user_train:
                    pass
                else:
                    user_train.append(user_id)
                verbose= user_id == chosen_verbose_user
                # perform regularization using generated samples after the first communication round
                user.train(
                    glob_iter,
                    user_id,
                    user_preds_matrix,
                    user_train,
                    #generator_model_dict,
                    personalized=self.personalized,
                    early_stop=self.early_stop,
                    verbose=verbose and glob_iter > 0,
                    regularization= glob_iter > 0 )
            curr_timestamp = time.time()  # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            #计算KL_散度    #计算出所有的user_id_2能教多少知识给user_id_1
            temp = 20
            soft_loss = nn.KLDivLoss(reduction="batchmean")
            kk_matrix = np.zeros((20, 20))
            #kk_matrix = np.zeros((25, 25))
            for user_id_1, user in zip(self.user_idxs, self.selected_users):
                student_preds = self.user_model_preds(user,user)
                #preds_matrix[user_id_1,user_id_1] = student_preds
                for user_id_2, users in zip(self.user_idxs, self.selected_users):
                    if user_id_1 == user_id_2:
                        pass
                    else:
                        teacher_preds = self.user_model_preds(user, users)
                        #preds_matrix[user_id_1, user_id_2] = teacher_preds
                        KL_ = soft_loss(
                            F.softmax(student_preds / temp, dim=1),  # softmax 是0—1的数 dim=1以行归一化：一行加起来等于1
                            F.softmax(teacher_preds / temp, dim=1))
                        kk_matrix[user_id_1, user_id_2] = 1 / -KL_.item()

            # 矩阵行的和为1   (矩阵（1，1） 是零点1)
            sum = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            k_matrix = np.zeros((20, 20))
            for i in range(20):
                for j in range(20):
                    if i == j:
                        pass
                    else:
                        sum[i] = sum[i] + kk_matrix[i,j]
            for i in range(20):
                if sum[i] == 0:
                    pass
                else:
                  for j in range(20):
                      if i == j:
                        k_matrix[i,j] = 0.1
                      else:
                        k_matrix[i,j] = kk_matrix[i,j] / sum[i] * 0.9

            '''#将用户的预测结果加权重聚合
            for user_id_1, user in zip(self.user_idxs, self.selected_users):
                i = 0
                for user_id_2, user in zip(self.user_idxs, self.selected_users):
                    if i==0:
                        user_preds_matrix[user_id_1] = preds_matrix[user_id_1,user_id_2] * k_matrix[user_id_1, user_id_2]
                    else:
                        user_preds_matrix[user_id_1] += preds_matrix[user_id_1,user_id_2]*k_matrix[user_id_1,user_id_2]
                    i += 1'''
            self.aggregate_parameters()       #平均聚合模型
            self.send_parameters(mode=self.mode) #给所有用户广播平均聚合模型
            self.evaluate()
            self.save_model() #写入全局模型信息，为了画图

            # 聚合个性化全局模型用于本地模型蒸馏
            for user_id, user in zip(self.user_idxs, self.selected_users):
                self.model_user = user
                self.per_aggregate_parameters(k_matrix,user_id)
                self.send_parameters(selected=True)


            '''for user_id, user in zip(self.user_idxs, self.selected_users):
                print(user_id)
                self.model_user = user
                self.per_aggregate_parameters(k_matrix,user_id)                  #带权重聚合模型
                #global_model_dict[user_id] = self.model.parameters()
                self.send_parameters(selected=True)  # 将模型广播给对应的用户
                self.evaluate_local_model(user)
                #self.save_model()
                #self.save_model(user_id)    #写入了每一个用户的信息
            self.save_model()  #仅仅写入了本轮最后一位客户的服务器端模型
            #self.evaluate_personalized_model()'''

            self.timestamp = time.time()  # log server-agg start time
            self.train_generator(
                self.batch_size,
                #k_matrix,
                #generator_model_dict,
                epoches=self.ensemble_epochs // self.n_teacher_iters,
                latent_layer_idx=self.latent_layer_idx,
                verbose=True
            )
            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
            if glob_iter  > 0 and glob_iter % 20 == 0 and self.latent_layer_idx == 0:
                self.visualize_images(self.generative_model, glob_iter, repeats=10)

        self.save_results(args)


    def train_generator(self, batch_size, epoches=1, latent_layer_idx=-1, verbose=False):
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'.
        :param batch_size:
        :param epoches:
        :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
        :param verbose: print loss information.
        :return: Do not return anything.
        """
        #self.generative_regularizer.train()
        self.label_weights, self.qualified_labels = self.get_label_weights()
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0

        def update_generator_(n_iters, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            self.generative_model.train()
            student_model.eval()
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                y=np.random.choice(self.qualified_labels, batch_size)        #从一个数组中随机抽取batch_size
                y_input=torch.LongTensor(y)                        # torch.LongTensor是64位整型 torch.tensor是一个类
                ## feed to generator
                gen_result=self.generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps=gen_result['output'], gen_result['eps']
                ##### get losses ####
                # decoded = self.generative_regularizer(gen_output)
                # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
                diversity_loss=self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs


                ######### get teacher loss ############
                teacher_loss=0
                teacher_logit=0
                for user_idx, user in enumerate(self.selected_users):
                    user.model.eval()
                    weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
                    expand_weight=np.tile(weight, (1, self.unique_labels))
                    user_result_given_gen=user.model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                    user_output_logp_=F.log_softmax(user_result_given_gen['logit'], dim=1)
                    teacher_loss_=torch.mean( \
                        self.generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                        torch.tensor(weight, dtype=torch.float32))
                    teacher_loss+=teacher_loss_
                    teacher_logit+=user_result_given_gen['logit'] * torch.tensor(expand_weight, dtype=torch.float32)

                ######### get student loss ############
                student_output=student_model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                student_loss=F.kl_div(F.log_softmax(student_output['logit'], dim=1), F.softmax(teacher_logit, dim=1))
                if self.ensemble_beta > 0:
                    loss=self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss
                else:
                    loss=self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += self.ensemble_alpha * teacher_loss#(torch.mean(TEACHER_LOSS.double())).item()
                STUDENT_LOSS += self.ensemble_beta * student_loss#(torch.mean(student_loss.double())).item()
                DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#(torch.mean(diversity_loss.double())).item()
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS

        for i in range(epoches):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS=update_generator_(
                self.n_teacher_iters, self.model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)

        TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        STUDENT_LOSS = STUDENT_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        info="Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        if verbose:
            print(info)
        self.generative_lr_scheduler.step()



    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):#26
            weights = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            for user_id,user in zip(self.user_idxs,self.selected_users):
                weights[user_id]=(user.label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append( np.array(weights) / np.sum(weights) )
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def visualize_images(self, generator, glob_iter, repeats=1):
        """
        Generate and visualize data for a generator.
        """
        os.system("mkdir -p images")
        path = f'images/{self.algorithm}-{self.dataset}-iter{glob_iter}.png'
        y=self.available_labels
        y = np.repeat(y, repeats=repeats, axis=0)
        y_input=torch.tensor(y)
        generator.eval()
        images=generator(y_input, latent=False)['output'] # 0,1,..,K, 0,1,...,K
        images=images.view(repeats, -1, *images.shape[1:])
        images=images.view(-1, *images.shape[2:])
        save_image(images.detach(), path, nrow=repeats, normalize=True)
        print("Image saved to {}".format(path))

