from FLAlgorithms.users.userFedProx import UserFedProx
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data,METRICS
# Implementation for FedProx Server
import copy
import torch
class FedProx(Server):
    def __init__(self, args, model, seed):
        #dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
        #         local_epochs, num_users, K, personal_learning_rate, times):
        super().__init__(args, model, seed)#dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         #local_epochs, num_users, times)
        self.metrics = {key: [] for key in METRICS}
        self.global_model = copy.deepcopy(self.model)
        # Initialize data for all  users
        data = read_data(args.dataset)
        total_users = len(data[0])
        print("Users in total: {}".format(total_users))

        for i in range(total_users):
            id, train_data , test_data = read_user_data(i, data, dataset=args.dataset)
            user = UserFedProx(args, id, model, train_data, test_data, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:", self.num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

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
        global_weights = []
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users = self.select_users(glob_iter,self.num_users)
            # average_acc = self.evaluate()
            local_weights, local_loss,local_acc = [], [],[]
            if glob_iter > 0:
                for user in self.users:
                    for global_model_param, user_model_param in zip(self.global_model.parameters(),
                                                                    user.model.parameters()):
                        if global_model_param.data.shape == user_model_param.data.shape:
                            user_model_param.data.copy_(global_model_param.data)
                        else:
                            raise ValueError("参数形状不匹配！")
            # self.send_parameters(mode=self.mode)
            # self.evaluate()
            for user in self.selected_users: # allow selected users to train
                    w, loss,acc = user.train(glob_iter,global_weights)
                    local_weights.append(copy.deepcopy(w))
                    local_loss.append(copy.deepcopy(loss))
                    local_acc.append(copy.deepcopy(acc))

            # self.aggregate_parameters()

            average_acc = sum(local_acc) / self.num_users
            average_loss = sum(local_loss) / len(local_loss)
            print("Average Accurancy = {:.4f}, Average Loss = {:.2f}.".format(average_acc, average_loss))


            global_weights = self.average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)


            # 评估模型
            loss, accuracy = self.global_model_evaluate(global_model=self.global_model)
            # loss, accuracy = self.global_model.evaluate(self.global_test_data)

            print("Global Accurancy = {:.4f}, Global Loss = {:.2f}.".format(-accuracy, loss))

            self.metrics['glob_acc'].append(-accuracy)
            self.metrics['glob_loss'].append(loss)
            self.metrics['average_acc'].append(average_acc)
            self.metrics['average_loss'].append(average_loss)
            # self.aggregate_parameters()


        self.save_results(args)
        self.save_model()