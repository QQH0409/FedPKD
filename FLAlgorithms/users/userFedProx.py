import torch
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import FedProxOptimizer

class UserFedProx(User):
    def __init__(self, args, id, model, train_data, test_data, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)

        self.optimizer = FedProxOptimizer(self.model.parameters(), lr=self.learning_rate, lamda=self.lamda)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter,global_weights, lr_decay=True, count_labels=False):
        self.clean_up_counts()
        # if glob_iter > 0:
        #     self.model.load_state_dict(global_weights)
        self.model.train()
        # cache global model initialized value to local model
        self.clone_model_paramenter(self.local_model, self.model.parameters())
        epoch_loss = []

        # 测试精度
        local_acc = 0
        # for x, y in self.test_data:
        loss = 0

        for i in range(1):
            for x, y in self.testloaderfull:
                # X, y = result['X'], result['y']
                output = self.model.get_outputs(x)['logit']
                output = self.model.soft_predict(output)
                output = output.type(torch.float)
                # y = torch.tensor(y, dtype=torch.float)
                y = y.type(torch.LongTensor)
                loss += self.loss(output, y)  # .long()
                # _loss = self.loss(output, y)
                # loss = loss + _loss
                local_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                len_y = len(y)

        local_acc = local_acc / len_y
        print('训练前：', local_acc)     #数量


        for epoch in range(self.local_epochs):
            self.model.train()
            batch_loss = []

            for i in range(self.K):
                result =self.get_next_train_batch(count_labels=count_labels)
                X, y = result['X'], result['y']
                if count_labels:
                    self.update_label_counts(result['labels'], result['counts'])

                self.optimizer.zero_grad()
                # output=self.model(X)['output']
                output = self.model.get_outputs(X)['logit']
                output = self.model.soft_predict(output)
                y = y.type(torch.LongTensor)
                loss=self.loss(output, y)
                loss.backward()
                self.optimizer.step(self.local_model)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if lr_decay:
            self.lr_scheduler.step(glob_iter)



        # 测试精度
        local_acc = 0
        # for x, y in self.test_data:
        loss = 0

        for i in range(1):
            for x, y in self.testloaderfull:
                # X, y = result['X'], result['y']
                output = self.model.get_outputs(x)['logit']
                output = self.model.soft_predict(output)
                output = output.type(torch.float)
                # y = torch.tensor(y, dtype=torch.float)
                y = y.type(torch.LongTensor)
                loss += self.loss(output, y)  # .long()
                # _loss = self.loss(output, y)
                # loss = loss + _loss
                local_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                len_y = len(y)

        local_acc = local_acc / len_y
        print('训练后：', local_acc)     #数量

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss),local_acc
