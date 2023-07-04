# -*- coding: utf-8 -*-
# @Time : 2022/9/26 19:57
# @Author : jklujklu
# @Email : jklujklu@126.com
# @File : dataset.py
# @Software: PyCharm

import os

import numpy as np
import pandas as pd
import pickle
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import mnist, cifar

from utils import load_image, dense_to_one_hot


class GetDataSet(object):
    def __init__(self, train_images_path, test_images_path='', is_iid=False):
        self.train_images_path = train_images_path
        self.test_images_path = test_images_path

        self.IMG_WIDTH = 30
        self.IMG_HEIGHT = 30

    def load_data(self):
        images = list()
        labels = list()
        NUM_CATEGORIES = len(os.listdir(self.train_images_path))
        for category in range(NUM_CATEGORIES):
            categories = os.path.join(self.train_images_path, str(category))
            for img in os.listdir(categories):
                img = load_image(os.path.join(categories, img), target_size=(self.IMG_WIDTH, self.IMG_HEIGHT))
                image = np.array(img)
                images.append(image)
                labels.append(dense_to_one_hot(category, num_classes=43))
        images = np.array(images)
        labels = np.array(labels)
        images = np.transpose(images, (0, 3, 1, 2))
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        x_train, x_test, y_train, y_test = train_test_split(np.array(images), labels, test_size=0.3)
        return x_train, x_test, y_train, y_test


class GetRoomDateset(object):
    def __init__(self, path):
        self.path = path
        pass

    def load_data(self):
        raw_data = pd.read_csv('/home/lhy/dataset/iot/Occupancy.csv')

        y = raw_data['Occupancy'].values
        X = raw_data.drop(['date', 'Occupancy'], axis=1).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test


class GetMNISTDataset(object):
    def __init__(self):
        pass

    @staticmethod
    def load_data():
        train_set = mnist.MNIST('./data', train=True, download=True)
        test_set = mnist.MNIST('./data', train=False, download=True)
        train_data, train_label = train_set.data, train_set.targets
        test_data, test_label = test_set.data, test_set.targets

        train_data = (np.array(train_data, dtype='float32') / 255)[:, np.newaxis, :]
        train_data = torch.from_numpy(train_data)

        test_data = (np.array(test_data, dtype='float32') / 255)[:, np.newaxis, :]
        test_data = torch.from_numpy(test_data)
        return train_data, train_label, test_data, test_label


class Cifar100(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self, dirname, train=True, start=-1, end=-1):
        super(Cifar100, self).__init__()
        if train:
            self.labels, self.images = self.load_cifar100(f"{dirname}/train")
        else:
            self.labels, self.images = self.load_cifar100(f"{dirname}/test")
        self.images = self.images.reshape(-1, 32, 32, 3)
        if start != -1 and end != -1 and train:
            self.labels = self.labels[start: end]
            self.images = self.images[start: end]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 默认的标准化参数
        ])
        image = transform(Image.fromarray(image))
        label = self.labels[index]
        label = int(label)
        return image, label

    def get_data(self):
        return self.images, self.labels

    @staticmethod
    def load_cifar100(path):
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            labels = data[b"fine_labels"]
            images = data[b"data"]
            return labels, images


if __name__ == '__main__':
    # d = GetDataSet('/home/lhy/dataset/GTSRB/Train', '')
    # print('a')
    a = Cifar100('/home/lhy/tmp/pycharm_project_867/normal/data/cifar-100-python', 0, 100)
    print(len(a))
    print('a')
