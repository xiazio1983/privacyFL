# -*- coding: utf-8 -*-
# @Time : 2022/9/26 21:46
# @Author : jklujklu
# @Email : jklujklu@126.com
# @File : server.py
# @Software: PyCharm
import json
import os
import sys
import time
import torch
from charm.toolbox.pairinggroup import PairingGroup
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from algorithm.sign import ShortSig
from dataset import GetDataSet, GetRoomDateset, GetMNISTDataset, Cifar100
from model import CNN, Linear, ConvNet, CIFAR100Model
from sample_way.client import BatchClient, ProxyClient
import torch.nn.functional as F

tic = None
bytes_length = 0
logger.add('log/mnist_cnn.log', rotation="50 MB", encoding='utf-8')


class BatchAggrServer:
    group = PairingGroup('SS512')

    def __init__(self):
        self.client_nums = 0  # 客户端数量
        self.aggregate = None  # 聚合模型
        self.global_model = []  # 全局模型
        self.lr = 0.01
        self.is_training = False
        self.sum_params = None
        self.nums = 4

        self.client_sids = []  # 客户端集合
        self.clients = {}
        self.ready_client_nums = 0
        self.proxy = None
        # 短签名算法相关参数
        self.sign_pub, self.sign_master, self.sign_gamma = ShortSig.param(self.group)
        # 客户端短签名私钥集
        self.sign_privs = {}

    def __init_global_model(self):
        print('load global model!')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        #
        # # CIFAR100
        # net = CIFAR100Model()
        # self.loss_func = F.cross_entropy
        # self.opti = optim.Adam(net.parameters(), lr=self.lr)
        # self.net = net.to(self.dev)

        # # GTSRB
        # net = CNN()
        # self.loss_func = F.cross_entropy
        # self.opti = optim.SGD(net.parameters(), lr=self.lr)
        # self.net = net.to(self.dev)

        # # Room Detect
        # net = Linear(5)
        # self.loss_func = F.binary_cross_entropy
        # self.opti = optim.Adam(net.parameters(), lr=0.01)
        # self.net = net.to(self.dev)

        # MNIST
        net = ConvNet()
        self.loss_func = F.cross_entropy
        self.opti = optim.Adam(net.parameters(), lr=0.01)
        self.net = net.to(self.dev)

        par = self.net.state_dict().copy()
        for key in par.keys():
            self.global_model.append(par[key].cpu().numpy())
        self.global_model = np.array(self.global_model)

    def __init_dataset(self):
        print('init dataset!')
        # cifar_train = Cifar100('/mnt/DEV_ST8000_01/code/pycharm_project_431/data/cifar-100-python', True)
        # cifar_test = Cifar100('/mnt/DEV_ST8000_01/code/pycharm_project_431/data/cifar-100-python', False)
        #
        # self.test_data_loader = DataLoader(cifar_test, batch_size=200,shuffle=False, drop_last=True)
        #
        # self.train_data,self.train_label = cifar_train.get_data()

        # # GTSRB
        # x_train, x_test, y_train, y_test = GetDataSet('/home/lhy/dataset/GTSRB/Train').load_data()
        # test_data = torch.tensor(x_test, dtype=torch.float32)
        # test_label = torch.argmax(torch.tensor(y_test, dtype=torch.float32), dim=1)
        # self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=64, shuffle=False,
        #                                    drop_last=True)
        #
        # x_train = torch.tensor(x_train, dtype=torch.float32)
        # y_train = torch.argmax(torch.tensor(y_train, dtype=torch.float32), dim=1)
        # self.train_data = x_train
        # self.train_label = y_train

        # # Room Detect
        # X_train, X_test, y_train, y_test = GetRoomDateset('/home/lhy/dataset/iot/Occupancy.csv').load_data()
        # X_train = torch.tensor(X_train, dtype=torch.float32)
        # y_train = torch.tensor(y_train, dtype=torch.float32)
        # X_test = torch.tensor(X_test, dtype=torch.float32)
        # y_test = torch.tensor(y_test, dtype=torch.float32)
        # self.test_data_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=True, drop_last=True)
        # self.train_data = X_train
        # self.train_label = y_train

        # MNIST
        X_train, y_train, X_test, y_test = GetMNISTDataset().load_data()
        self.test_data_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=True, drop_last=True)
        self.train_data = X_train
        self.train_label = y_train

    def __init_clients(self):
        print('init clients!')
        max_num = len(self.train_data)
        self.num_interval = int(max_num / self.nums)
        # self.num_interval = 500
        # 为所有用户创建签名私钥
        users = []
        for i in range(self.nums + 1):
            sk, pk, A = ShortSig.gen(self.group, self.sign_pub, self.sign_gamma)
            users.append((sk, pk, A))

        # 创建代理客户端
        self.proxy = ProxyClient((self.sign_pub, users[0]))

        # 创建训练客户端
        for i in range(self.nums):
            # 分配训练样本
            images = self.train_data[self.num_interval * i: self.num_interval * (i + 1)]
            labels = self.train_label[self.num_interval * i: self.num_interval * (i + 1)]
            train_dl = DataLoader(TensorDataset(images, labels), batch_size=64, shuffle=True, drop_last=True)
            # 创建训练客户端
            self.client_sids.append(f'client_{i}')
            self.clients.update(
                {f'client_{i}': BatchClient(f'client_{i}', train_dl, (self.sign_pub, users[i + 1]), self.proxy)})

    def start(self):
        logger.info('start training!')
        self.__init_global_model()
        self.__init_dataset()
        self.__init_clients()

        z1s = None
        for i in range(10):
            logger.info(f'round {i}')

            # 根据全局模型获取模型每一层的最大最小值，用于缩放本地模型
            scaled_params = []
            par = self.net.state_dict().copy()
            for key in par.keys():
                _ = par[key].cpu().numpy()
                _max, _min = _.max(), _.min()
                scaled_params.append((_max, _min))

            sum_params = 0  # 梯度累加
            signs = []  # 签名数组
            for client in self.clients.keys():
                en_msg, sign = self.clients[client].local_train(self.global_model, 0.01, 5, scaled_params, z1s=z1s,
                                                                nums=self.nums)
                self.clients[client].local_val(self.test_data_loader)
                sum_params += en_msg
                signs.append(sign)

            tic = time.time()
            # 批量验证签名是否有效
            if self.proxy.batch_verify(signs):
                print(f'batch verify used: {int((time.time() - tic) * 1000)}ms')
                tic = time.time()
                # 梯度解密
                dec_gradients = self.proxy.decrypt(sum_params)
                print(f'dec_gradients used: {int((time.time() - tic) * 1000)}ms')
                # 梯度还原
                tic = time.time()
                unscaled_params = []
                for gradients, scaled_param in zip(dec_gradients, scaled_params):
                    _max, _min = scaled_param
                    unscaled_params.append(gradients / self.nums / 1024 * (_max - _min) + _min)
                print(f'unscaled_params used: {int((time.time() - tic) * 1000)}ms')
                # 更新全局模型
                self.global_model = np.array(unscaled_params)
            else:
                logger.error('Batch Verify Failed!!!')
                break
            # # 全局模型评估
            # par = self.net.state_dict().copy()
            # for key, param in zip(self.net.state_dict().keys(), self.global_model):
            #     par[key] = torch.from_numpy(param)
            # self.net.load_state_dict(par, strict=True)
            # sum_accu = 0
            # num = 0
            # for data, label in self.test_data_loader:
            #     data, label = data.to(self.dev), label.to(self.dev)
            #     preds = self.net(data)
            #     preds = torch.argmax(preds, dim=1)
            #     sum_accu += (preds == label).float().mean()
            #     num += 1
            # print(f'acc: {sum_accu / num}')


if __name__ == '__main__':
    BatchAggrServer().start()
