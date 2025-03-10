from FLAlgorithms.users.userFedpkd import UserFedpkd
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
import gc
MIN_SAMPLES_PER_LABEL=1

class Fedpkd(Server):
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
        self.global_model = copy.deepcopy(self.model)
        self.generative_model = create_generative_model(args.dataset, args.algorithm, self.model_name, args.embedding)
        if not args.train:
            print('number of generator parameteres: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = self.generative_model.latent_layer_idx
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
            user=UserFedpkd(
                args, id, model, self.generative_model,
                train_data, test_data,
                self.available_labels, self.latent_layer_idx, label_info,
                use_adam=self.use_adam)
            self.users.append(user)
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedGen server.")

    def average_weights(self,w):
        """
        average the weights from all local models
        """
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg

    def train(self, args):
        global_features = {}
        global_soft_prediction = {}
        #### pretraining
        for glob_iter in range(self.num_glob_iters):       #200次循环迭代
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            # 选择此轮训练的用户——在main.py中
            self.selected_users, self.user_idxs=self.select_users(glob_iter, self.num_users, return_idx=True)
            local_weights = []
            # 每一轮训练都要广播
            if not self.local:
                if glob_iter >0:
                    for user_id, user in zip(self.user_idxs, self.users):
                        for global_model_param, user_model_param in zip(self.global_model.parameters(),
                                                                        user.model.parameters()):
                            if global_model_param.data.shape == user_model_param.data.shape:
                                user_model_param.data.copy_(global_model_param.data)
                            else:
                                raise ValueError("参数形状不匹配！")
                # self.send_parameters(mode=self.mode)# broadcast averaged prediction model

            self.evaluate()
            chosen_verbose_user = np.random.randint(0, len(self.users))  #np.random.randint返回一个随机用户
            self.timestamp = time.time() # log user-training start time


            for user_id, user in zip(self.user_idxs, self.selected_users): # allow selected users to train
                verbose= user_id == chosen_verbose_user
                # perform regularization using generated samples after the first communication round
                w = user.train(
                    glob_iter,
                    global_features=global_features,
                    global_soft_prediction=global_soft_prediction,
                    personalized=self.personalized,
                    early_stop=self.early_stop,
                    verbose=verbose and glob_iter > 0,
                    regularization= glob_iter > 0
                )
                local_weights.append(copy.deepcopy(w))
                # fedhkd的原型学习
                # local_features, local_soft_predictions = user.generate_knowledge(
                #     temp=0.5)
                # global_features.update(local_features)
                # global_soft_prediction.update(local_soft_predictions)
                # del local_features
                # del local_soft_predictions
                # gc.collect()
            curr_timestamp = time.time() # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            # 是个性化本地模型 计算每个的本地模型 含pFed
            if self.personalized:
                self.evaluate_personalized_model()

            self.timestamp = time.time() # log server-agg start time

            # self.train_generator(
            #     self.batch_size,
            #     epoches=self.ensemble_epochs // self.n_teacher_iters,
            #     latent_layer_idx=self.latent_layer_idx,
            #     verbose=True
            # )

            # update global weights
            global_weights = self.average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)

            # self.aggregate_parameters()
            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
            if glob_iter  > 0 and glob_iter % 20 == 0 and self.latent_layer_idx == 0:
                self.visualize_images(self.generative_model, glob_iter, repeats=10)

        self.save_results(args)
        self.save_model()

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
            # self.generative_model.to('cuda')
            self.generative_model.train()
            student_model.eval()
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                y=np.random.choice(self.qualified_labels, batch_size)       #从一个数组中随机抽取batch_size
                y_input=torch.LongTensor(y)  #.to('cuda')                     # torch.LongTensor是64位整型 torch.tensor是一个类
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
                    user_result_given_gen=user.model.get_outputs(gen_output)['logit']
                    user_result_given_gen = user.model.soft_predict(user_result_given_gen)
                    user_output_logp_=F.log_softmax(user_result_given_gen, dim=1)
                    teacher_loss_=torch.mean( \
                        self.generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                        torch.tensor(weight, dtype=torch.float32))
                    teacher_loss+=teacher_loss_
                    teacher_logit+=user_result_given_gen['logit'] * torch.tensor(expand_weight, dtype=torch.float32)

                ######### get student loss ############
                student_output=student_model.classifier(gen_output)
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
        for label in range(self.unique_labels):
            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])
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
