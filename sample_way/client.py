# -*- coding: utf-8 -*-
# @Time : 2022/9/26 21:46
# @Author : jklujklu
# @Email : jklujklu@126.com
# @File : client.py
# @Software: PyCharm
import pickle
import time
import torch
from charm.core.math.pairing import ZR
from charm.toolbox.pairinggroup import PairingGroup
from loguru import logger
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F

from algorithm.sign import ShortSig
from dataset import GetDataSet
from model import CNN, Linear, ConvNet, CIFAR100Model


class BatchClient(object):
    def __init__(self, cid, train_dl, sign_params, proxy):
        self.net = None
        self.opti = None
        self.loss_func = None
        self.dev = None
        self.id = cid
        self.train_dl = train_dl
        self.group = PairingGroup('SS512')
        self.sign_pub_key, self.sign_priv_key = sign_params

        self.proxy = proxy

    def __sign(self):
        return ShortSig.sign(self.group, self.sign_pub_key, self.sign_priv_key)

    def encrypt(self, gradients):
        tic = time.time()
        secret = ShortSig.agree(self.sign_priv_key[1], self.proxy.get_public_key())
        print(f'KA.agree used {int((time.time() - tic) * 1000)}ms')
        mask = int(self.group.hash(secret.__str__(), ZR))
        return gradients + mask

    def sign(self, data):
        tic = time.time()
        msg = self.__sign()
        print(f'sign priv used {int((time.time() - tic) * 1000)}ms')
        tic = time.time()
        ShortSig.sign2(self.group, self.sign_pub_key, self.sign_priv_key, data)
        print(f'sign data used {int((time.time() - tic) * 1000)}ms')
        return msg

    def local_train(self, params, lr, epoch, scale_params, z1s=None, nums=4, ):
        # # GTSRB
        # self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # net = CNN()
        # self.loss_func = F.cross_entropy
        # self.opti = optim.SGD(net.parameters(), lr=lr)
        # # Room Detect
        # self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # net = Linear(5)
        # self.loss_func = F.binary_cross_entropy
        # self.opti = optim.Adam(net.parameters(), lr=0.01)
        # MNIST
        net = ConvNet()
        # net = CIFAR100Model()
        self.loss_func = F.cross_entropy
        self.opti = optim.Adam(net.parameters(), lr=0.01)
        self.net = net.to(self.dev)

        # tic = time.time()
        # if z1s is not None:
        #     new_params = []
        #     su = self.group.init(0)
        #     for i in z1s:
        #         su += self.encrypt_priv_key[4] * i
        #     for param in params:
        #         if type(param - su) == np.ndarray:
        #             _ = ((param - su).astype(np.int) / nums / 1024 * 5 - 2.5)
        #         elif type(param - su) == int:
        #             _ = ((param - su) / nums / 1024 * 5 - 2.5)
        #         new_params.append(_)
        #     params = np.array(new_params)

        par = net.state_dict().copy()
        for key, param in zip(par.keys(), params):
            par[key] = torch.from_numpy(param.astype(np.float64))
        net.load_state_dict(par, strict=True)
        self.net = net.to(self.dev)
        # print(f'dec used {int((time.time() - tic) * 1000)}ms')

        for epoch in range(epoch):
            train_loss = 0
            batches = 0
            sum_accu = 0
            num = 0
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                # data = data.type(torch.cuda.FloatTensor)
                preds = net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                self.opti.step()
                self.opti.zero_grad()
                train_loss += loss.item()
                batches += 1
                _, preds = torch.max(preds.data, 1)
                # preds[preds >= 0.5] = 1
                # preds[preds < 0.5] = 0

                sum_accu += (preds == label).float().mean()
                num += 1

            logger.debug('epoch: {} | Loss: {:.3f} | Acc: {:.3f}'.
                         format(epoch, train_loss / (batches + 1), sum_accu / num))

        tic = time.time()
        params = []
        par = self.net.state_dict().copy()
        for key, scale in zip(par.keys(), scale_params):
            _ = par[key].cpu().numpy()
            _max, _min = scale
            params.append(np.around((_ - _min) / (_max - _min) * 1024).astype(np.int))
        print(f'gradients scale used {int((time.time() - tic) * 1000)}ms')
        tic = time.time()
        eng_msg = self.encrypt(np.array(params), )
        print(f'gradients enc used {int((time.time() - tic) * 1000)}ms')
        tic = time.time()
        sign = self.sign(np.array(params))
        print(f'gradients sign used {int((time.time() - tic) * 1000)}ms')
        return eng_msg, sign

    def local_val(self, test_dl):
        self.net.eval()
        sum_accu = 0
        num = 0
        for data, label in test_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = self.net(data)
            # # GTSRB
            # preds = torch.argmax(preds, dim=1)

            # Room Detect
            # preds[preds >= 0.5] = 1
            # preds[preds < 0.5] = 0

            # # MNIST
            _, preds = torch.max(preds.data, 1)

            sum_accu += (preds == label).float().mean()
            num += 1
        logger.info(f'\tClient: {self.id} local acc -> {sum_accu / num}')


class ProxyClient(object):
    def __init__(self, sign_params):
        self.group = PairingGroup('SS512')
        self.sign_pub_key, self.sign_priv_key = sign_params
        self.clients = []

    def batch_verify(self, msgs):
        self.clients = [msg[1] for msg in msgs]
        return ShortSig.batch_verify(self.group, self.sign_pub_key, msgs)

    def decrypt(self, gradients):
        for i in self.clients:
            secret = ShortSig.agree(self.sign_priv_key[1], i)
            gradients -= int(self.group.hash(secret.__str__(), ZR))
        return gradients

    def get_public_key(self):
        return self.sign_priv_key[2]


if __name__ == '__main__':
    group = PairingGroup('SS512')
    gpk, gmsk, gamma = ShortSig.param(group)
    users = []
    for i in range(3):
        sk, pk, A = ShortSig.gen(group, gpk, gamma)
        users.append((sk, pk, A))

    proxy = ProxyClient((gpk, users[0]))
    client_a = BatchClient('a', None, (gpk, users[1]), proxy)
    client_b = BatchClient('b', None, (gpk, users[2]), proxy)

    f1 = open('../aa.pickle', 'rb')
    params = np.array(pickle.load(f1))
    scaled_gradients = []
    scaled_params = []
    for param in params:
        _max, _min = param.max(), param.min()
        scaled_gradients.append(np.around((param - _min) / (_max - _min) * 1024).astype(np.int))
        scaled_params.append((_max, _min))
    scaled_gradients = np.array(scaled_gradients)

    gra_1 = client_a.encrypt(scaled_gradients)
    sign1 = client_a.sign(gra_1)
    gra_2 = client_b.encrypt(scaled_gradients)
    sign2 = client_b.sign(gra_2)

    print(proxy.batch_verify([sign1, sign2]))
    dec_gradients = proxy.decrypt(gra_1 + gra_2)

    unscaled_params = []
    for gradients, scaled_param in zip(dec_gradients, scaled_params):
        _max, _min = scaled_param
        unscaled_params.append(gradients / 2 / 1024 * (_max - _min) + _min)

    print('aaa')
