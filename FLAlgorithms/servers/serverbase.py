import torch
import os
import numpy as np
import h5py
from utils.model_utils import get_dataset_name, RUNCONFIGS
import copy
import torch.nn.functional as F
import time
import torch.nn as nn
from utils.model_utils import get_log_path, METRICS

class Server:
    def __init__(self, args, model, seed):
        # Set up the main attributes
        self.dataset = args.dataset
        self.num_glob_iters = args.num_glob_iters
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.total_train_samples = 0
        self.K = args.K
        self.model = copy.deepcopy(model[0])
        self.model_name = model[1]
        self.users = []
        self.selected_users = []
        self.num_users = args.num_users
        self.beta = args.beta
        self.lamda = args.lamda
        self.algorithm = args.algorithm
        self.personalized = 'pFed' in self.algorithm
        self.mode='partial' if 'partial' in self.algorithm.lower() else 'all'
        self.seed = seed
        self.deviations = {}
        self.metrics = {key:[] for key in METRICS}
        self.timestamp = None
        self.save_path = args.result_path
        os.system("mkdir -p {}".format(self.save_path))


    def init_ensemble_configs(self):
        #### used for ensemble learning ####
        dataset_name = get_dataset_name(self.dataset)
        self.ensemble_lr = RUNCONFIGS[dataset_name].get('ensemble_lr', 1e-4)
        self.ensemble_batch_size = RUNCONFIGS[dataset_name].get('ensemble_batch_size', 128)
        self.ensemble_epochs = RUNCONFIGS[dataset_name]['ensemble_epochs']
        self.num_pretrain_iters = RUNCONFIGS[dataset_name]['num_pretrain_iters']
        self.temperature = RUNCONFIGS[dataset_name].get('temperature', 1)
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.ensemble_alpha = RUNCONFIGS[dataset_name].get('ensemble_alpha', 1)
        self.ensemble_beta = RUNCONFIGS[dataset_name].get('ensemble_beta', 0)
        self.ensemble_eta = RUNCONFIGS[dataset_name].get('ensemble_eta', 1)
        self.weight_decay = RUNCONFIGS[dataset_name].get('weight_decay', 0)
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']
        self.ensemble_train_loss = []
        self.n_teacher_iters = 5
        self.n_student_iters = 1
        print("ensemble_lr: {}".format(self.ensemble_lr) )
        print("ensemble_batch_size: {}".format(self.ensemble_batch_size) )
        print("unique_labels: {}".format(self.unique_labels) )


    def if_personalized(self):
        return 'pFed' in self.algorithm or 'PerAvg' in self.algorithm

    def if_ensemble(self):
        return 'FedE' in self.algorithm

    #计算在学生模型数据下的模型测试user的数据，users的模型
    def user_model_preds(self,user,users):
        user_model_result = user.get_result(users)
        return user_model_result


    def send_parameters(self, mode='all', beta=1, selected=False):
        users = self.users
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.model_user
            if mode == 'all': # share only subset of parameters   #更换新旧
                users.set_teacher_model_parameters(self.model,beta=beta)
            else: # share all parameters
                users.set_shared_parameters(self.model,mode=mode)
        else:
            for user in users:
               if mode == 'all': # share only subset of parameters   #更换新旧
                  user.set_parameters(self.model,beta=beta)
               else: # share all parameters
                  user.set_shared_parameters(self.model,mode=mode)


    def add_parameters(self, user, ratio, partial=False):
        if partial:
            for server_param, user_param in zip(self.model.get_shared_parameters(), user.model.get_shared_parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        else:
            for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio

    #原来是def aggregate_parameters(self, partial=False):
    def aggregate_parameters(self,partial=False):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        if partial:
            for param in self.model.get_shared_parameters():
                param.data = torch.zeros_like(param.data)
        else:
            for param in self.model.parameters():
                param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train,partial=partial)

    def per_add_parameters(self, user,k_matrix,model_user_id,user_id, ratio, partial=False):
        if partial:
            for server_param, user_param in zip(self.model.get_shared_parameters(), user.model.get_shared_parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        else:
            for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
            #server_param.data = server_param.data + k_matrix[model_user_id, user_id] * user_param.data.clone() * ratio
                server_param.data = server_param.data + k_matrix[model_user_id,user_id]* user_param.data.clone() * ratio
    # 个性化的模型参数累加
    def per_aggregate_parameters(self,k_matrix,model_user_id,partial=False):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        if partial:
            for param in self.model.get_shared_parameters():
                param.data = torch.zeros_like(param.data)
        else:
            for param in self.model.parameters():
                param.data = torch.zeros_like(param.data)
        total_train = 0
        user = self.model_user
        total_train += user.train_samples
        for user_id, users in zip(self.user_idxs, self.selected_users):
            self.per_add_parameters(user, k_matrix,model_user_id,user_id,users.train_samples / total_train,partial=partial)



    '''def save_model(self,model_user_id):
        if self.algorithm == 'pFed_DataFreeKD':
            model_path = os.path.join("models", self.dataset, str(model_user_id))
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(self.model, os.path.join(model_path, "server" + ".pt"))
        else:
            model_path = os.path.join("models", self.dataset)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(self.model, os.path.join(model_path, "server" + ".pt"))'''

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))


    def per_save_model(self,moder_user_id):
        model_path = os.path.join("models", self.dataset,str(moder_user_id))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)


    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users, return_idx=False):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        if return_idx:
            user_idxs = np.random.choice(range(len(self.users)), num_users, replace=False)  # , p=pk)
            return [self.users[i] for i in user_idxs], user_idxs
        else:
            return np.random.choice(self.users, num_users, replace=False)


    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")#,log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()


    def save_results(self, args):
        alg = get_log_path(args, args.algorithm, self.seed, args.gen_batch_size)
        with h5py.File("./{}/{}.h5".format(self.save_path, alg), 'w') as hf:
            for key in self.metrics:
                hf.create_dataset(key, data=self.metrics[key])
            hf.close()


    def test(self, selected=False):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses


    def test_local_model(self,user,selected=True):
        '''tests self.latest_model on given clients
        '''
        num_samples = []          # 总样本
        tot_correct = []                # 对的
        losses = []
        ct, ns, loss = user.test_personalized_model()
        tot_correct.append(ct*1.0)             # A.append(b) 在列表A的末尾加上b
        num_samples.append(ns)
        losses.append(loss)
        ids = [user.id]
        return ids, num_samples, tot_correct, losses


    def test_personalized_model(self, selected=True):
        '''tests self.latest_model on given clients
        '''
        num_samples = []          # 总样本
        tot_correct = []                # 对的
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, ns, loss = c.test_personalized_model()
            tot_correct.append(ct*1.0)             # A.append(b) 在列表A的末尾加上b
            num_samples.append(ns)
            losses.append(loss)
        ids = [c.id for c in self.users]
        return ids, num_samples, tot_correct, losses

    def test_per_user_model(self, selected=True):
        '''tests self.latest_model on given clients
        '''
        num_samples = []          # 总样本
        tot_correct = []           # 对的
        losses = []
        for user_id, user in zip(self.user_idxs, self.selected_users):
            ct, ns, loss = user.test_personalized_model(user_id)
            tot_correct.append(ct*1.0)             # A.append(b) 在列表A的末尾加上b
            num_samples.append(ns)
            losses.append(loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def user_model_evaluate(self, selected=True, save=True):
        stats = self.test_per_user_model(selected=selected)
        test_ids, test_num_samples, test_tot_correct, test_losses = stats[:4]
        glob_acc = np.sum(test_tot_correct)*1.0/np.sum(test_num_samples)
        test_loss = np.sum([x * y.detach().numpy() for (x, y) in zip(test_num_samples, test_losses)]).item() / np.sum(test_num_samples)
        if save:
            self.metrics['per_acc'].append(glob_acc)
            self.metrics['per_loss'].append(test_loss)
        print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(glob_acc, test_loss))

    def evaluate_personalized_model(self, selected=True, save=True):
        stats = self.test_personalized_model(selected=selected)
        test_ids, test_num_samples, test_tot_correct, test_losses = stats[:4]
        glob_acc = np.sum(test_tot_correct)*1.0/np.sum(test_num_samples)
        test_loss = np.sum([x * y.detach().numpy() for (x, y) in zip(test_num_samples, test_losses)]).item() / np.sum(test_num_samples)
        if save:
            self.metrics['per_acc'].append(glob_acc)
            self.metrics['per_loss'].append(test_loss)
        print("Average 222Global Accurancy = {:.4f}, Loss = {:.2f}.".format(glob_acc, test_loss))

    def evaluate_local_model(self, user,selected=True, save=True):
        stats = self.test_local_model(user,selected=selected)
        test_ids, test_num_samples, test_tot_correct, test_losses = stats[:4]
        glob_acc = np.sum(test_tot_correct)*1.0/np.sum(test_num_samples)
        test_loss = np.sum([x * y.detach().numpy() for (x, y) in zip(test_num_samples, test_losses)]).item() / np.sum(test_num_samples)
        if save:
            self.metrics['per_acc'].append(glob_acc)
            self.metrics['per_loss'].append(test_loss)
        print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(glob_acc, test_loss))


    def evaluate_ensemble(self, selected=True):
        self.model.eval()
        users = self.selected_users if selected else self.users
        test_acc=0
        loss=0
        for x, y in self.testloaderfull:
            target_logit_output=0
            for user in users:
                # get user logit
                user.model.eval()
                user_result=user.model(x, logit=True)
                target_logit_output+=user_result['logit']
            target_logp=F.log_softmax(target_logit_output, dim=1)
            test_acc+= torch.sum( torch.argmax(target_logp, dim=1) == y ) #(torch.sum().item()
            y = y.type(torch.LongTensor)
            loss+=self.loss(target_logp, y)
        loss = loss.detach().numpy()
        test_acc = test_acc.detach().numpy() / y.shape[0]
        self.metrics['glob_acc'].append(test_acc)
        self.metrics['glob_loss'].append(loss)
        print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(test_acc, loss))


    def evaluate(self, save=True, selected=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test(selected=selected)
        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
        #glob_loss = np.sum([x * y for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
        # glob_loss = np.sum([x* y.detach().numpy() for (x, y) in zip(test_samples, test_losses)]) / np.sum(test_samples)

        # if save:
        #     self.metrics['glob_acc'].append(glob_acc)
        #     self.metrics['glob_loss'].append(glob_loss)
        # print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))

        return glob_acc

    def global_model_evaluate(self, global_model):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.global_model_test(global_model)
        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
        #glob_loss = np.sum([x * y for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
        glob_loss = np.sum([x* y.detach().cpu().numpy() for (x, y) in zip(test_samples, test_losses)]) / np.sum(test_samples)

        # if save:
        #     self.metrics['glob_acc'].append(glob_acc)
        #     self.metrics['glob_loss'].append(glob_loss)
        # print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))

        return glob_acc, glob_loss

    def global_model_test(self, global_model):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users
        for c in users:
            ct, c_loss, ns = c.global_model_test(global_model)
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

