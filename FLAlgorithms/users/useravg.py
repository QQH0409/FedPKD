import torch
from FLAlgorithms.users.userbase import User

class UserAVG(User):
    def __init__(self,  args, id, model, train_data, test_data, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.test_data = test_data

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count



    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter,global_weights, personalized=False, lr_decay=True, count_labels=True):
        if glob_iter > 0:
            self.model.load_state_dict(global_weights)
        self.clean_up_counts()
        self.model.train().to('cuda')
        # local_acc = 0
        # # for x, y in self.test_data:
        # loss = 0
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
                loss=self.ce(output, y)            #loss
                loss.backward()
                self.optimizer.step()#self.plot_Celeb)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))



            #不理解
            # # local-model <=== self.model
            # self.clone_model_paramenter(self.model.parameters(), self.local_model)
            # if personalized:
            #     self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
            # # local-model ===> self.model
            # #self.clone_model_paramenter(self.local_model, self.model.parameters())



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
