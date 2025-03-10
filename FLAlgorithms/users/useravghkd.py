import torch
from FLAlgorithms.users.userbase import User
import copy
import gc

class UserAVGhkd(User):
    def __init__(self,  args, id, model, train_data, test_data, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def dict_to_tensor(self, dic):
        lit = []
        for key,tensor in dic.items():
            lit.append(tensor[0])
        lit = torch.stack(lit)
        return lit
    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter,global_weights,global_features, global_soft_prediction, personalized=False, lr_decay=True, count_labels=True):
        self.clean_up_counts()
        if glob_iter > 0:
            self.model.load_state_dict(global_weights)
        self.model.train().to('cuda')

        local_acc = 0
        # for x, y in self.test_data:
        loss = 0
        # for i in range(1):
        #     for x, y in self.testloaderfull:
        #         # X, y = result['X'], result['y']
        #         x = x.to('cuda')
        #         output = self.model.get_outputs(x)['logit'].to('cuda')
        #         output = self.model.soft_predict(output).to('cuda')
        #         output = output.type(torch.float).to('cuda')
        #         # y = torch.tensor(y, dtype=torch.float)
        #         y = y.type(torch.LongTensor).to('cuda')
        #         loss += self.loss(output, y)  # .long()
        #         # _loss = self.loss(output, y)
        #         # loss = loss + _loss
        #         local_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        #         len_y = len(y)
        #
        # local_acc = local_acc / len_y
        # print('训练前：', local_acc)  # 数量


        epoch_loss = []
        for epoch in range(1, self.local_epochs + 1):
            batch_loss = []
            self.model.train().to('cuda')
            for i in range(self.K):
                result =self.get_next_train_batch(count_labels=count_labels)
                X, y = result['X'].to('cuda'), result['y'].to('cuda')
                if count_labels:
                    self.update_label_counts(result['labels'], result['counts'])

                self.optimizer.zero_grad()
                output=self.model.get_outputs(X)['logit'].to('cuda')
                output=self.model.soft_predict(output).to('cuda')
                y = y.type(torch.LongTensor).to('cuda')
                loss=self.ce(output, y)

                #### 原型学习 #######
                if glob_iter > 0:

                    tensor_global_features = self.dict_to_tensor(global_features).to('cuda')
                    tensor_global_soft_prediction = self.dict_to_tensor(global_soft_prediction).to('cuda')


                    F_out = self.model.get_outputs(X)['features'].to('cuda')
                    Z = self.model.get_outputs(X)['logit'].to('cuda')
                    Z = self.model.soft_predict(Z).to('cuda')      #归一化的logit 也就是10个类
                    # Z_help = self.model.classifier(tensor_global_features).to('cuda')# 全局模型的logits输出
                    Z_help = self.model.classifier(tensor_global_features).to('cuda')  # 全局模型的logits输出
                    Q_help = self.model.soft_predict(Z_help, temp=0.5 )  # 全局模型的软测试
                    loss1 = self.ce(Z, y)  # 本地模型交叉熵损失               #ce
                    target_features = copy.deepcopy(F_out.data)

                    for i in range(y.shape[0]):
                        if int(y[i]) in global_features.keys():
                            target_features[i] = global_features[int(y[i])][0].data

                    target_features = target_features.to('cuda')
                    if len(global_features) == 0:
                        loss2 = 0 * loss1
                        loss3 = 0 * loss1
                    else:
                        loss2 = -self.kld(Q_help, tensor_global_soft_prediction)  # 分类器提取器   #Z_help
                        loss3 = 0
                        # loss3 = self.mse(F_out, target_features)  # 特征提取器
                        # loss2 = 0 * loss1
                        # loss3 = 0 * loss1
                    # print('Loss1: {:.6f} Loss2: {:.6f}  Loss3: {:.6f} '.format(loss1.item(), loss2.item(), loss3.item()))

                    loss = loss + loss2 + loss3




                loss.backward()
                self.optimizer.step()#self.plot_Celeb)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # local-model <=== self.model
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
            if personalized:
                self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
            # local-model ===> self.model
            #self.clone_model_paramenter(self.local_model, self.model.parameters())
        if lr_decay:
            self.lr_scheduler.step(glob_iter)

            # 测试精度
            local_acc = 0
            # for x, y in self.test_data:
            loss = 0

            for i in range(1):
                for x, y in self.testloaderfull:
                    # X, y = result['X'], result['y']
                    x = x.to('cuda')
                    output = self.model.get_outputs(x)['logit'].to('cuda')
                    output = self.model.soft_predict(output).to('cuda')
                    output = output.type(torch.float).to('cuda')
                    # y = torch.tensor(y, dtype=torch.float)
                    y = y.type(torch.LongTensor).to('cuda')
                    loss += self.loss(output, y)  # .long()
                    # _loss = self.loss(output, y)
                    # loss = loss + _loss
                    local_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    len_y = len(y)

            local_acc = local_acc / len_y
            # print('训练后：', local_acc)     #数量

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss),local_acc


    def generate_knowledge(self, temp):
        self.model.to('cuda')
        self.model.eval().to('cuda')
        local_features = {}
        local_soft_prediction = {}
        # num_classes = self.model.module.num_classes
        # num_classes = 10
        # features = [torch.zeros(self.code_length).to('cuda')] * num_classes
        # soft_predictions = [torch.zeros(num_classes).to('cuda')] * num_classes
        # count = [0] * num_classes
        for batch_idx, (X, y) in enumerate(self.trainloader):
            # F = self.model.module.feature_extractor(X)
            X = X.to('cuda')
            F = self.model.feature_extractor(X).to('cuda')
            Z = self.model.get_outputs(X)['logit'].to('cuda')
            Q = self.model.soft_predict(Z,temp=0.5).to('cuda')

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
            gc.collect()

        features, soft_predictions = self.local_knowledge_aggregation(local_features, local_soft_prediction,std=2)       #std of gaussian noise

        return (features, soft_predictions)

    def local_knowledge_aggregation(self, local_features, local_soft_prediction, std):
        agg_local_features = dict()
        agg_local_soft_prediction = dict()
        # feature_noise = std * torch.randn(64).to('cuda')     #self.args.code_len
        for [label, features] in local_features.items():
            if len(features) > 1:
                feature = 0 * features[0].data.to('cuda')
                for i in features:
                    feature += i.data.to('cuda')
                # agg_local_features[label] = [feature / len(features) + feature_noise]
                agg_local_features[label] = [feature / len(features) ]
            else:
                # agg_local_features[label] = [features[0].data + feature_noise]
                agg_local_features[label] = [features[0].data]

        for [label, soft_prediction] in local_soft_prediction.items():
            if len(soft_prediction) > 1:
                soft = 0 * soft_prediction[0].data.to('cuda')
                for i in soft_prediction:
                    soft += i.data.to('cuda')

                agg_local_soft_prediction[label] = [soft / len(soft_prediction)]
            else:
                agg_local_soft_prediction[label] = [soft_prediction[0].data]

        return agg_local_features, agg_local_soft_prediction