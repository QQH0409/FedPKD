import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from utils.model_utils import get_dataset_name
from utils.model_config import RUNCONFIGS
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(
            self, args, id, model, train_data, test_data, use_adam=False):
        self.model = copy.deepcopy(model[0])
        self.teacher_model = copy.deepcopy(model[0])
        self.model_name = model[1]
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.local_epochs = args.local_epochs
        self.algorithm = args.algorithm
        self.K = args.K
        self.dataset = args.dataset
        self.ce = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss()
        self.mse = nn.MSELoss()
        #self.trainloader = DataLoader(train_data, self.batch_size, drop_last=False)
        self.trainloader = DataLoader(train_data, self.batch_size, shuffle=False, drop_last=True)    #原来是shuffle=True
        self.testloader =  DataLoader(test_data, self.batch_size, drop_last=False)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        dataset_name = get_dataset_name(self.dataset)
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']

        # those parameters are for personalized federated learning.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.per_local_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.prior_decoder = None
        self.prior_params = None

        self.init_loss_fn()
        if use_adam:
            self.optimizer=torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.learning_rate, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=1e-2, amsgrad=False)
        else:
            self.optimizer = pFedIBOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        self.label_counts = {}


    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def get_user_per_generative_model_parameters(self,user_id, generator_model_dict ,beta=1):
        for old_param, new_param in zip(self.generative_model.parameters(), generator_model_dict[user_id]):
            if beta == 1:
                old_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
    def get_user_per_parameters(self,user_id, global_model_dict ,beta=1):
        for old_param, new_param, local_param in zip(self.model.parameters(), global_model_dict[user_id], self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
                local_param.data = beta * new_param.data.clone() + (1-beta) * local_param.data.clone()

    def set_teacher_model_parameters(self, model,beta=1):
        for old_param, new_param in zip(self.teacher_model.parameters(), model.parameters()):
            if beta == 1:
                old_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
    #将用户模型更新为全局模型的参数
    def set_parameters(self, model,beta=1):

        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
                local_param.data = beta * new_param.data.clone() + (1-beta) * local_param.data.clone()


    def set_prior_decoder(self, model, beta=1):
        for new_param, local_param in zip(model.personal_layers, self.prior_decoder):
            if beta == 1:
                local_param.data = new_param.data.clone()
            else:
                local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()


    def set_prior(self, model):
        for new_param, local_param in zip(model.get_encoder() + model.get_decoder(), self.prior_params):
            local_param.data = new_param.data.clone()

    # only for pFedMAS
    def set_mask(self, mask_model):
        for new_param, local_param in zip(mask_model.get_masks(), self.mask_model.get_masks()):
            local_param.data = new_param.data.clone()

    def get_result(self,users):
        samples = self.get_next_train_batch(count_labels=True)
        X, y = samples['X'], samples['y']
        self.update_label_counts(samples['labels'], samples['counts'])
        model_result = users.model(X, logit=True)
        output = model_result['output']
        return output



    def set_shared_parameters(self, model, mode='decode'):
        # only copy shared parameters to local
        for old_param, new_param in zip(
                self.model.get_parameters_by_keyword(mode),
                model.get_parameters_by_keyword(mode)
        ):
            old_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()


    def clone_model_paramenter(self, param, clone_param):
        with torch.no_grad():
            for param, clone_param in zip(param, clone_param):
                clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params, keyword='all'):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def evaluate_model(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        for x, y in self.testloaderfull:
            output = self.model(x)['output']
            #y = torch.tensor(y, dtype=torch.float)
            #output = torch.tensor(output, dtype=torch.float)
            y = y.type(torch.LongTensor)
            loss += self.loss(output, y)    #.long()
            #_loss = self.loss(output, y)
            #loss = loss + _loss
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        #glob_acc = np.sum(test_acc) * 1.0 / np.sum(test_samples)
        # glob_loss = np.sum([x * y for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
        #glob_loss = np.sum([x * y.detach().numpy() for (x, y) in zip(test_samples, test_losses)]) / np.sum(test_samples)
        test_acc = test_acc / y.shape[0]
        return  loss, test_acc

    def test(self):
        self.model.eval().to('cuda')
        test_acc = 0
        loss = 0
        for x, y in self.testloaderfull:
            x = x.to('cuda')
            output = self.model.get_outputs(x)['logit'].to('cuda')
            output = self.model.soft_predict(output).to('cuda')
            output = output.type(torch.float).to('cuda')
            #y = torch.tensor(y, dtype=torch.float)
            y = y.type(torch.LongTensor).to('cuda')
            loss += self.loss(output, y)    #.long()
            #_loss = self.loss(output, y)
            #loss = loss + _loss
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return test_acc, loss, y.shape[0]              #ct, c_loss, ns

    def global_model_test(self,global_model):
        global_model.eval().to('cuda')
        test_acc = 0
        loss = 0
        for x, y in self.testloaderfull:
            x = x.to('cuda')
            output = global_model.get_outputs(x)['logit'].to('cuda')
            output = global_model.soft_predict(output).to('cuda')
            output = output.type(torch.float).to('cuda')
            #y = torch.tensor(y, dtype=torch.float)
            y = y.type(torch.LongTensor).to('cuda')
            loss += self.loss(output, y)    #.long()
            #_loss = self.loss(output, y)
            #loss = loss + _loss
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return test_acc, loss, y.shape[0]              #ct, c_loss, ns





    def test_personalized_model(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(self.personalized_model_bar)
        for x, y in self.testloaderfull:
            # output = self.model(x)['output']
            logit = self.model.get_outputs(x)['logit']
            output = self.model.soft_predict(logit)
            y = y.type(torch.LongTensor)
            loss += self.loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0], loss


    def get_next_train_batch(self, count_labels=True):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        result = {'X': X, 'y': y}
        if count_labels:
            unique_y, counts=torch.unique(y, return_counts=True)
            unique_y = unique_y.detach().numpy()
            counts = counts.detach().numpy()
            result['labels'] = unique_y
            result['counts'] = counts
        return result

    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self,user_id):
        if self.algorithm == 'pFed_DataFreeKD':
            model_path = os.path.join("models", self.dataset, str(user_id))
            self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
        else:
            model_path = os.path.join("models", self.dataset)
            self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    def per_load_model(self,user_id):
        model_path = os.path.join("models", self.dataset,str(user_id))
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))


