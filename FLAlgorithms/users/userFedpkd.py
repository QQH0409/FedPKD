import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
from FLAlgorithms.users.userbase import User
from utils.utils import Accuracy,soft_predict
from FLAlgorithms.trainmodel.models import FeatureExtractor
import gc
import copy

class UserFedpkd(User):
    def __init__(self,
                 args, id, model, generative_model,
                 train_data, test_data,
                 available_labels, latent_layer_idx, label_info,
                 use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.gen_batch_size = args.gen_batch_size
        self.generative_model = generative_model
        self.latent_layer_idx = latent_layer_idx
        self.available_labels = available_labels
        self.label_info=label_info
        self.ce = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss()
        self.mse = nn.MSELoss()


    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:1 for label in range(self.unique_labels)}

    def dict_to_tensor(self, dic):
        lit = []
        for key,tensor in dic.items():
            lit.append(tensor[0])
        lit = torch.stack(lit)
        return lit

    def train(self, glob_iter,global_features, global_soft_prediction, personalized=False, early_stop=100, regularization=True, verbose=False):
        self.clean_up_counts()
        self.model.train()
        self.generative_model.eval()


        #optimizer = nn.SGD(params=net.trainable_params(), learning_rate=0.01)
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0
        for epoch in range(self.local_epochs):
            self.model.train()
            for i in range(self.K):

                result = self.get_next_train_batch(count_labels=True)
                X, y = result['X'], result['y']

                self.update_label_counts(result['labels'], result['counts'])

                #### sample from real dataset (un-weighted)

                self.optimizer.zero_grad()


                output=self.model.get_outputs(X)['logit']
                output = self.model.soft_predict(output)
                y = y.type(torch.LongTensor)
                loss = -self.loss(output, y)      # 硬标签预测损失predictive_loss

                #optimizer = 1

                #### sample y and generate z
                # if regularization and epoch < early_stop and glob_iter > 0:
                #     generative_alpha=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                #     generative_beta=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)
                #     ### get generator output(latent representation) of the same label
                #     gen_output=self.generative_model(y, latent_layer_idx=self.latent_layer_idx)['output']
                #     logit_given_gen=self.model.classifier(gen_output)
                #     target_p=F.softmax(logit_given_gen['logit'], dim=1).clone().detach()
                #     user_latent_loss= generative_beta * self.ensemble_loss(user_output_logp, target_p)
                #
                #
                #     sampled_y=np.random.choice(self.available_labels, self.gen_batch_size)
                #     #原来是sampled_y=torch.tensor(sampled_y)
                #     sampled_y = torch.LongTensor(sampled_y)
                #     gen_result=self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx)
                #     gen_output=gen_result['output'] # latent representation when latent = True, x otherwise
                #     user_output_logp =self.model.classifier(gen_output)['logit']
                #     user_output_logp = self.model.soft_predict(user_output_logp)
                #     teacher_loss =  generative_alpha * torch.mean(
                #         self.generative_model.crossentropy_loss(user_output_logp, sampled_y)
                #     )
                #     # this is to further balance oversampled down-sampled synthetic data
                #     gen_ratio = self.gen_batch_size / self.batch_size
                #     loss=predictive_loss + gen_ratio * teacher_loss + user_latent_loss
                #     TEACHER_LOSS+=teacher_loss
                #     LATENT_LOSS+=user_latent_loss
                # else:
                #     #### get loss and perform optimization
                #     loss=predictive_loss

                #### 原型学习 #######
                # if regularization and epoch < early_stop and glob_iter > 0:
                #     tensor_global_features = self.dict_to_tensor(global_features)
                #     tensor_global_soft_prediction = self.dict_to_tensor(global_soft_prediction)
                #
                #     F_out = self.model.get_outputs(X)['features']
                #     Z = self.model.get_outputs(X)['logit']
                #     Z = self.model.soft_predict(Z)      #归一化的logit 也就是10个类
                #     Z_help = self.model.classifier(tensor_global_features)# 全局模型的logits输出
                #     # Q_help = soft_predict(tensor_global_features, temp=0.5 )  # 全局模型的软测试
                #     loss1 = self.ce(Z, y)  # 本地模型交叉熵损失
                #     target_features = copy.deepcopy(F_out.data)
                #
                #     for i in range(y.shape[0]):
                #         if int(y[i]) in global_features.keys():
                #             target_features[i] = global_features[int(y[i])][0].data
                #
                #     target_features = target_features
                #     if len(global_features) == 0:
                #         loss2 = 0 * loss1
                #         loss3 = 0 * loss1
                #     else:
                #         loss2 = self.kld(Z_help, tensor_global_soft_prediction)  # 分类器提取器
                #         loss3 = self.mse(F_out, target_features)  # 特征提取器

                    # if epoch == 0:
                    #     print('loss:', loss.item(),'loss2:', loss2.item(),'loss3:', loss3.item())

                    #loss = loss + loss2 + loss3


                loss.backward()
                self.optimizer.step()#self.local_model)
        # local-model <=== self.model
        # 克隆模型参数
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        if personalized:
            self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        self.lr_scheduler.step(glob_iter)
        # 随机挑一个用户看他本地的损失函数
        # if regularization and verbose:
        #     TEACHER_LOSS=TEACHER_LOSS / (self.local_epochs * self.K)                #.detach().numpy()
        #     LATENT_LOSS=LATENT_LOSS / (self.local_epochs * self.K)              #.detach().numpy()
        #     info='\nUser Teacher Loss={:.4f}'.format(TEACHER_LOSS)
        #     info+=', Latent Loss={:.4f}'.format(LATENT_LOSS)
        #     print(info)
        return self.model.state_dict()
    def generate_knowledge(self, temp):
        self.model
        self.model.eval()
        local_features = {}
        local_soft_prediction = {}
        # num_classes = self.model.module.num_classes
        # num_classes = 10
        # features = [torch.zeros(self.code_length).to('cuda')] * num_classes
        # soft_predictions = [torch.zeros(num_classes).to('cuda')] * num_classes
        # count = [0] * num_classes
        for batch_idx, (X, y) in enumerate(self.trainloader):
            # F = self.model.module.feature_extractor(X)
            F = self.model.feature_extractor(X)
            Z = self.model.get_outputs(X)['logit']
            Q = self.model.soft_predict(Z,temp=0.5)

            # m = y.shape[0]
            for i in range(len(y)):
                if y[i].item() in local_features:
                    local_features[y[i].item()].append(F[i, :])
                    local_soft_prediction[y[i].item()].append(Q[i, :])
                else:
                    local_features[y[i].item()] = [F[i, :]]
                    local_soft_prediction[y[i].item()] = [Q[i, :]]
            del X
            del y
            del F
            del Z
            del Q
            # gc.collect()

        features, soft_predictions = self.local_knowledge_aggregation(local_features, local_soft_prediction,std=2)       #std of gaussian noise

        return (features, soft_predictions)

    def local_knowledge_aggregation(self, local_features, local_soft_prediction, std):
        agg_local_features = dict()
        agg_local_soft_prediction = dict()
        # feature_noise = std * torch.randn(64).to('cuda')     #self.args.code_len
        for [label, features] in local_features.items():
            if len(features) > 1:
                feature = 0 * features[0].data
                for i in features:
                    feature += i.data
                # agg_local_features[label] = [feature / len(features) + feature_noise]
                agg_local_features[label] = [feature / len(features) ]
            else:
                # agg_local_features[label] = [features[0].data + feature_noise]
                agg_local_features[label] = [features[0].data]

        for [label, soft_prediction] in local_soft_prediction.items():
            if len(soft_prediction) > 1:
                soft = 0 * soft_prediction[0].data
                for i in soft_prediction:
                    soft += i.data

                agg_local_soft_prediction[label] = [soft / len(soft_prediction)]
            else:
                agg_local_soft_prediction[label] = [soft_prediction[0].data]

        return agg_local_features, agg_local_soft_prediction



    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples['counts']
        #weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
        np_y = samples['y'].detach().numpy()
        n_labels = samples['y'].shape[0]
        weights = np.array([n_labels / count for count in counts]) # smaller count --> larger weight
        weights = len(self.available_labels) * weights / np.sum(weights) # normalized
        label_weights = np.ones(self.unique_labels)
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]
        return sample_weights


#优化器自定义
# class MomentumOpt(nn.Optimizer):
#     def __init__(self, params, learning_rate, momentum, weight_decay=0.0, loss_scale=1.0, use_nesterov=False):
#         super(MomentumOpt, self).__init__(learning_rate, params, weight_decay, loss_scale)
#         self.momentum = Parameter(Tensor(momentum, mstype.float32), name="momentum")
#         self.moments = self.parameters.clone(prefix="moments", init="zeros")
#         self.opt = ops.ApplyMomentum(use_nesterov=use_nesterov)
#         self.assign = ops.Assign()
#     def construct(self, gradients):
#         params = self.parameters
#         moments = self.moments
#         success = None
#         for param, mom, grad in zip(params, moments, gradients):
#             # update = self.momentum * param + mom + self.learning_rate * grad
#             # success = self.assign(param, update)
#             success = self.opt(param, mom, self.learning_rate, grad, self.momentum)
#         return success
