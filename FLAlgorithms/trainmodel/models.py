import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_config import CONFIGS_

import collections

#################################
##### Neural Network model #####
#################################
# class Net(nn.Module):
#     def __init__(self, dataset='mnist', model='cnn'):      #这是类的构造函数
#         super(Net, self).__init__()                         # 调用父类 nn.Module 的构造函数,进行初始化。
#         # define network layers
#         print("Creating model for {}".format(dataset))
#         self.dataset = dataset
#         configs, input_channel, self.output_dim, self.hidden_dim, self.latent_dim=CONFIGS_[dataset]
#         print('Network configs:', configs)
#         self.named_layers, self.layers, self.layer_names =self.build_network(
#             configs, input_channel, self.output_dim)
#         self.n_parameters = len(list(self.parameters()))   #计算模型的总参数数量,并将其赋给 self.n_parameters 属性。
#         self.n_share_parameters = len(self.get_encoder())  #计算编码器部分的参数数量,并将其赋给 self.n_share_parameters 属性
#
#     def get_number_of_parameters(self):    #定义了一个方法 get_number_of_parameters,用于获取模型的总参数数量
#         pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
#         return pytorch_total_params
#
#     def build_network(self, configs, input_channel, output_dim):
#         layers = nn.ModuleList()
#         named_layers = {}
#         layer_names = []
#         kernel_size, stride, padding = 3, 2, 1
#         for i, x in enumerate(configs):
#             if x == 'F':
#                 layer_name='flatten{}'.format(i)
#                 layer=nn.Flatten(1)
#                 layers+=[layer]
#                 layer_names+=[layer_name]
#             elif x == 'M':
#                 pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
#                 layer_name = 'pool{}'.format(i)
#                 layers += [pool_layer]
#                 layer_names += [layer_name]
#             else:
#                 cnn_name = 'encode_cnn{}'.format(i)
#                 cnn_layer = nn.Conv2d(input_channel, x, stride=stride, kernel_size=kernel_size, padding=padding)
#                 named_layers[cnn_name] = [cnn_layer.weight, cnn_layer.bias]
#
#                 bn_name = 'encode_batchnorm{}'.format(i)
#                 bn_layer = nn.BatchNorm2d(x)
#                 named_layers[bn_name] = [bn_layer.weight, bn_layer.bias]
#
#                 relu_name = 'relu{}'.format(i)
#                 relu_layer = nn.ReLU(inplace=True)# no parameters to learn
#
#                 layers += [cnn_layer, bn_layer, relu_layer]
#                 layer_names += [cnn_name, bn_name, relu_name]
#                 input_channel = x
#
#         # finally, classification layer
#         fc_layer_name1 = 'encode_fc1'
#         fc_layer1 = nn.Linear(self.hidden_dim, self.latent_dim)
#         layers += [fc_layer1]
#         layer_names += [fc_layer_name1]
#         named_layers[fc_layer_name1] = [fc_layer1.weight, fc_layer1.bias]
#
#         fc_layer_name = 'decode_fc2'
#         fc_layer = nn.Linear(self.latent_dim, self.output_dim)
#         layers += [fc_layer]
#         layer_names += [fc_layer_name]
#         named_layers[fc_layer_name] = [fc_layer.weight, fc_layer.bias]
#         return named_layers, layers, layer_names
#
#
#     def get_parameters_by_keyword(self, keyword='encode'):
#         params=[]
#         for name, layer in zip(self.layer_names, self.layers):
#             if keyword in name:
#                 #layer = self.layers[name]
#                 params += [layer.weight, layer.bias]
#         return params
#
#     def get_encoder(self):
#         return self.get_parameters_by_keyword("encode")
#
#     def get_decoder(self):
#         return self.get_parameters_by_keyword("decode")
#
#     def get_shared_parameters(self, detach=False):
#         return self.get_parameters_by_keyword("decode_fc2")
#
#     def get_learnable_params(self):
#         return self.get_encoder() + self.get_decoder()
#
#     def forward(self, x, start_layer_idx = 0, logit=False):
#         """
#         :param x:
#         :param logit: return logit vector before the last softmax layer
#         :param start_layer_idx: if 0, conduct normal forward; otherwise, forward from the last few layers (see mapping function)
#         :return:
#         """
#         if start_layer_idx < 0: #
#             return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
#         restults={}
#         z = x
#         for idx in range(start_layer_idx, len(self.layers)):
#             layer_name = self.layer_names[idx]
#             layer = self.layers[idx]
#             z = layer(z)
#
#         if self.output_dim > 1:
#             restults['output'] = F.log_softmax(z, dim=1)
#         else:
#             restults['output'] = z
#         if logit:
#             restults['logit']=z
#         return restults
#
#     def mapping(self, z_input, start_layer_idx=-1, logit=True):
#         z = z_input
#         n_layers = len(self.layers)
#         for layer_idx in range(n_layers + start_layer_idx, n_layers):
#             layer = self.layers[layer_idx]
#             z = layer(z)
#         if self.output_dim > 1:
#             out=F.log_softmax(z, dim=1)
#         result = {'output': out}
#         if logit:
#             result['logit'] = z
#         return result

class Net(nn.Module):
    def __init__(self, dataset='emnist', model='cnn'):
        super(Net, self).__init__()

        # 特征提取器
        if model == 'cnn':
            # self.feature_extractor = nn.Sequential(
            #     nn.Conv2d(1, 32, 3, 1),
            #     nn.ReLU(),
            #     nn.Conv2d(32, 64, 3, 1),
            #     nn.ReLU(),
            #     nn.MaxPool2d(2),
            #     nn.Dropout2d(0.25),
            #     nn.Flatten(),
            #     nn.Linear(9216, 32),             #这个是mnist
            #     nn.ReLU(),
            #     nn.Dropout(0.5)
            # )
            if dataset == 'mnist':
                input_channels = 1
                num_classes = 10
                # 考虑到MNIST的简单性，使用较简单的网络结构
                # self.feature_extractor = nn.Sequential(
                #     nn.Conv2d(input_channels, 32, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.Conv2d(32, 64, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.MaxPool2d(2),
                #     nn.Dropout2d(0.25),
                #     nn.Flatten(),
                #     nn.Linear(14 * 14 * 64, 128),  # 假设输入图像是28x28，经过两次卷积和一次池化后，大小变为7x7 #7 * 7 * 64
                #     nn.ReLU(),
                #     nn.Dropout(0.5)
                # )
                #更复杂一点的网络
                self.feature_extractor = nn.Sequential(
                    # 第一层卷积，32个3x3的卷积核，保持空间大小不变
                    nn.Conv2d(input_channels, 32, 3, 1, padding=1),  # 加深：从32增加到64
                    nn.ReLU(),
                    nn.BatchNorm2d(32),  # 添加BatchNorm帮助训练
                    # 第二层卷积，128个3x3的卷积核，保持空间大小不变
                    nn.Conv2d(32, 64, 3, 1, padding=1),  # 加深加宽：通道数从64增加到128
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    # 池化层，空间大小减半
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25),
                    # 第三层卷积，继续加深加宽，256个3x3的卷积核
                    nn.Conv2d(64, 128, 3, 1, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(128),
                    # 再次池化层，空间大小减半
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25),
                    # 展平操作
                    nn.Flatten(),
                    # 全连接层，输入特征数量取决于卷积和池化后的空间大小（此处假设是4x4）
                    nn.Linear(7 * 7 * 128, 256),  # 假设空间大小变为4x4，并加宽全连接层到512个神经元   #4 * 4 * 256
                    nn.ReLU(),
                    nn.Dropout(0.5),

                )
            elif dataset == 'emnist':
                input_channels = 1
                num_classes = 26  # EMNIST根据具体子集可能有所不同，这里假设是字母子集
                # 相对于MNIST，EMNIST可能需要更深或更宽的网络来捕捉更多细节
                # self.feature_extractor = nn.Sequential(
                #     nn.Conv2d(input_channels, 64, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.Conv2d(64, 128, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.MaxPool2d(2),
                #     nn.Conv2d(128, 256, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.MaxPool2d(2),
                #     nn.Dropout2d(0.25),
                #     nn.Flatten(),
                #     nn.Linear(7 * 7 * 256, 512),  # 假设输入图像是较大尺寸的，如32x32，这里的大小需要调整
                #     nn.ReLU(),
                #     nn.Dropout(0.5)
                # )

                # 跟mnist一样，（多加一层GPU不足）
                self.feature_extractor = nn.Sequential(
                    # 第一层卷积，32个3x3的卷积核，保持空间大小不变
                    nn.Conv2d(input_channels, 32, 3, 1, padding=1),  # 加深：从32增加到64
                    nn.ReLU(),
                    nn.BatchNorm2d(32),  # 添加BatchNorm帮助训练
                    # 第二层卷积，128个3x3的卷积核，保持空间大小不变
                    nn.Conv2d(32, 64, 3, 1, padding=1),  # 加深加宽：通道数从64增加到128
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    # 池化层，空间大小减半
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25),
                    # 第三层卷积，继续加深加宽，256个3x3的卷积核
                    nn.Conv2d(64, 128, 3, 1, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(128),

                    # 再次池化层，空间大小减半
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25),
                    # 展平操作
                    nn.Flatten(),
                    # 全连接层，输入特征数量取决于卷积和池化后的空间大小（此处假设是4x4）
                    nn.Linear(7 * 7 * 128, 256),  # 假设空间大小变为4x4，并加宽全连接层到512个神经元   #4 * 4 * 256
                    nn.ReLU(),
                    nn.Dropout(0.5),

                )

                # 更复杂一点的网络  GPU内存不足
                # self.feature_extractor = nn.Sequential(
                #     # 第一层卷积块
                #     nn.Conv2d(input_channels, 64, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.BatchNorm2d(64),
                #
                #     # 第二层卷积块，加宽通道数
                #     nn.Conv2d(64, 128, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.BatchNorm2d(128),
                #     nn.MaxPool2d(2),  # 空间大小减半
                #     nn.Dropout2d(0.25),
                #
                #     # 第三层卷积块，继续加宽
                #     nn.Conv2d(128, 256, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.BatchNorm2d(256),
                #     nn.Conv2d(256, 256, 3, 1, padding=1),  # 额外的卷积层以增加深度
                #     nn.ReLU(),
                #     nn.BatchNorm2d(256),
                #     nn.MaxPool2d(2),  # 空间大小再次减半
                #     nn.Dropout2d(0.25),
                #
                #     # 第四层卷积块，进一步加宽
                #     nn.Conv2d(256, 512, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.BatchNorm2d(512),
                #     nn.Conv2d(512, 512, 3, 1, padding=1),  # 额外的卷积层
                #     nn.ReLU(),
                #     nn.BatchNorm2d(512),
                #     nn.MaxPool2d(2),  # 空间大小再减半
                #     nn.Dropout2d(0.25),
                #
                #     # 展平操作
                #     nn.Flatten(),
                #
                #     # 全连接层，输入特征数量取决于卷积和池化后的空间大小
                #     # 假设经过三次池化后，空间大小从28x28变为7x7（这取决于实际网络结构和EMNIST图像尺寸）
                #     # 然后乘以最终的通道数（这里是512）
                #     nn.Linear(3 * 3 * 512, 1024),  # 假设空间大小变为7x7，并加宽全连接层到1024个神经元
                #     nn.ReLU(),
                #     nn.Dropout(0.5),
                # )
            elif dataset == 'celeb':
                input_channels = 3  # CelebA是彩色图像
                num_classes = 2  # 假设是二分类任务，如识别是否微笑等
                # 对于更复杂的CelebA数据集，需要更深的网络结构和更多的参数
                # self.feature_extractor = nn.Sequential(
                #     nn.Conv2d(input_channels, 64, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.Conv2d(64, 128, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.MaxPool2d(2),
                #     nn.Conv2d(128, 256, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.Conv2d(256, 512, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.MaxPool2d(2),
                #     nn.Conv2d(512, 512, 3, 1, padding=1),
                #     nn.ReLU(),
                #     nn.MaxPool2d(2),
                #     nn.Dropout2d(0.25),
                #     nn.Flatten(),
                #     nn.Linear(7 * 7 * 512, 1024),  # 假设输入图像是较大尺寸的，这里的大小需要调整
                #     nn.ReLU(),
                #     nn.Dropout(0.5)
                # )
                # 跟mnist一样，（多加一层GPU不足）
                self.feature_extractor = nn.Sequential(
                    # 第一层卷积，32个3x3的卷积核，保持空间大小不变
                    nn.Conv2d(input_channels, 32, 3, 1, padding=1),  # 加深：从32增加到64
                    nn.ReLU(),
                    nn.BatchNorm2d(32),  # 添加BatchNorm帮助训练
                    # 第二层卷积，128个3x3的卷积核，保持空间大小不变
                    nn.Conv2d(32, 64, 3, 1, padding=1),  # 加深加宽：通道数从64增加到128
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    # 池化层，空间大小减半
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25),
                    # 第三层卷积，继续加深加宽，256个3x3的卷积核
                    nn.Conv2d(64, 128, 3, 1, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(128),

                    # 再次池化层，空间大小减半
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25),
                    # 展平操作
                    nn.Flatten(),
                    # 全连接层，输入特征数量取决于卷积和池化后的空间大小（此处假设是4x4）
                    nn.Linear(21 * 21 * 128, 256),  # 假设空间大小变为4x4，并加宽全连接层到512个神经元
                    nn.ReLU(),
                    nn.Dropout(0.5),

                )
            else:
                raise ValueError("Invalid dataset. Choose 'mnist', 'emnist', or 'celeba'.")
            self.classifier = nn.Sequential(
                nn.Linear(256, num_classes))

        elif model == 'mlp':
            # self.feature_extractor = nn.Sequential(
            #     nn.Flatten(),
            #     nn.Linear(784, 32),
            #     nn.ReLU(),
            #     nn.Dropout(0.5)
            # )
            #并不是隐藏层越多越大模型就就好，很容易过拟合。
            if dataset == 'mnist':
                num_classes = 10
                self.feature_extractor = nn.Sequential(
                    nn.Flatten(),  # 假设输入是[batch_size, 1, 28, 28]或类似，展平为[batch_size, 784]    yii
                    nn.Linear(784, 200),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    # nn.Linear(200, 100),


                )
            elif dataset == 'emnist' :
                num_classes = 26
                print('111111111111111111111111111111111111')
                self.feature_extractor = nn.Sequential(
                    nn.Flatten(),  # 假设输入是[batch_size, 1, 28, 28]或类似，展平为[batch_size, 784]    yii
                    nn.Linear(784, 200),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    # nn.Linear(200, 100),


                )
            elif dataset == 'celeb':   #没有调整参数，论文对比用不上
                num_classes = 2
                self.feature_extractor = nn.Sequential(
                    nn.Flatten(),  # 假设输入是[batch_size, 1, 28, 28]或类似，展平为[batch_size, 784]    yii
                    nn.Linear(21168, 1000),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1000, 200),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                )
            self.classifier = nn.Sequential(
                nn.Linear(200, num_classes))

        else:
            raise ValueError("Invalid model type. Choose 'cnn' or 'mlp'.")




    def forward(self, X):
        f = self.feature_extractor(X)
        Z = self.classifier(f)
        return f, Z

    def get_outputs(self, X):
        # 使用 forward 方法获取特征 f 和分类结果 Z
        f, Z = self.forward(X)
        # 创建一个字典来存储输出
        outputs = {'features': f, 'logit': Z}
        return outputs

    def soft_predict(self, Z, temp=1.0):
        return F.softmax(Z / temp, dim=1)    #  归一化的output

    def evaluate(self, global_test_data):
        self.eval()  # 将模型设置为评估模式
        total = 0
        correct = 0
        loss_sum = 0

        with torch.no_grad():  # 不计算梯度，减少内存消耗和加速计算
            for inputs, targets in global_test_data:
                inputs, targets = torch.tensor(inputs), torch.tensor(targets)
                outputs = F.softmax(self.forward(inputs)['logit'])
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # 假设我们使用的是交叉熵损失函数
                loss = F.cross_entropy(outputs, targets)
                loss_sum += loss.item()
                print(total)

        accuracy = 100 * correct / total
        average_loss = loss_sum / len(global_test_data)
        return average_loss, accuracy

class EncoderFemnist(nn.Module):
    def __init__(self, code_length):
         super(EncoderFemnist, self).__init__()
         self.conv1 = nn.Conv2d(3, 6, 5)
         self.pool = nn.MaxPool2d(2, 2)
         self.conv2 = nn.Conv2d(6, 10, 5)
         self.fc1 = nn.Linear(int(250), code_length)
         # self.fc1 = nn.Linear(16 * 5 * 5, 120)
         # self.fc2 = nn.Linear(120, 84)
         # self.fc3 = nn.Linear(84, args.num_classes)
    #     self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
    #     self.conv2 = nn.Conv2d(10,20, kernel_size=5)
    #     self.conv2_drop = nn.Dropout2d()
    #     self.fc1 = nn.Linear(int(500), code_length)        #320
    #
    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     # x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
    #     x = x.view(-1, 500)
    #     z = F.relu(self.fc1(x))
    #     return z

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 5)
        z = F.relu(self.fc1(x))
        return z


class CNNFemnist(nn.Module):
    def __init__(self, args, code_length=64, num_classes=10):
        super(CNNFemnist, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = EncoderFemnist(self.code_length)
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(self.code_length, self.num_classes),
                                        nn.LogSoftmax(dim=1))

    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return p

class FeatureExtractor(nn.Module):
    def __init__(self, net):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*net.layers[:-2]) # 除去最后的全连接层

    def forward(self, x):
        return self.features(x)
