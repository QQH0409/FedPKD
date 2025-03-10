# （FedGen-main）2023/03/08 10:38
from FLAlgorithms.users.user_Localtrainning import Local
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data,METRICS
import numpy as np
# Implementation for FedAvg Server
import time
import copy

class LOCALtra(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)
        self.metrics = {key: [] for key in METRICS}


        # Initialize data for all  users
        data = read_data(args.dataset)       #data包括clients, groups, train_data, test_data,proxy_data
        total_users = 100      #客户端总数
        self.local_model = {}
        for i in range (total_users):
            self.local_model[i] = copy.deepcopy(self.model)
        self.use_adam = 'adam' in self.algorithm.lower()
        print("Users in total: {}".format(total_users))

        for i in range(total_users):
            id, train_data , test_data = read_user_data(i, data, dataset=args.dataset)
            user = Local(args, id, model, train_data, test_data, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Number of users / total users:",args.num_users, " / " ,total_users)
        print("Finished creating Local training server.")

    def train(self, args):
        local_model_weights = []
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            # self.selected_users = self.select_users(glob_iter,self.num_users)
            self.selected_users, self.user_idxs = self.select_users(glob_iter, self.num_users, return_idx=True)
            # self.send_parameters(mode=self.mode)
            local_acc, local_loss = [],[]
            # self.evaluate()
            self.timestamp = time.time() # log user-training start time

            # for user in self.selected_users: # allow selected users to train
            for user_id, user in zip(self.user_idxs, self.selected_users):
                # print(user_id)
                if glob_iter > 0:
                    local_model_weights = self.local_model[user_id].state_dict()

                model_weights, loss,acc = user.train(glob_iter, local_model_weights, personalized=self.personalized) #* user.train_samples

                self.local_model[user_id].load_state_dict(model_weights)
                local_loss.append(copy.deepcopy(loss))
                local_acc.append(copy.deepcopy(acc))
            curr_timestamp = time.time() # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)

            average_acc = sum(local_acc) / self.num_users
            average_loss = sum(local_loss) / len(local_loss)
            print("Average Accurancy = {:.4f}, Average Loss = {:.2f}.".format(average_acc, average_loss))

            self.metrics['average_acc'].append(average_acc)
            self.metrics['average_loss'].append(average_loss)

            # Evaluate selected user
            if self.personalized:
                # Evaluate personal model on user for each iteration
                print("Evaluate personal model\n")
                self.evaluate_personalized_model()

            self.timestamp = time.time() # log server-agg start time
            # self.aggregate_parameters()
            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
        self.save_results(args)
        self.save_model()