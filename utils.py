# -*- coding: utf-8 -*-
# @Time : 2022/9/26 16:27
# @Author : jklujklu
# @Email : jklujklu@126.com
# @File : utils.py
# @Software: PyCharm
from charm.core.math.pairing import pc_element
import pickle
import os

from PIL import Image
import numpy as np


def serialize(group, element):
    if type(element) == tuple or type(element) == list:
        results = []
        for _ in element:
            if type(_) == pc_element:
                results.append(group.serialize(_).hex())
            else:
                results.append(_)
        return results
    elif type(element) == dict:
        results = element.copy()
        for key in results.keys():
            if type(element[key]) == pc_element:
                results[key] = group.serialize(element[key]).hex()
        return results


def deserialize(group, element):
    if type(element) == tuple or type(element) == list:
        results = []
        for _ in element:
            results.append(group.deserialize(bytes.fromhex(_)))
        return results
    elif type(element) == dict:
        results = element.copy()
        for key in results.keys():
            results[key] = group.deserialize(bytes.fromhex(element[key]))
        return results


def weights_encoding(x):
    return pickle.dumps(x)


def weights_decoding(s):
    return pickle.loads(s)


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def load_image(path, target_size=None):
    img = Image.open(path)
    if target_size:
        img = img.resize(target_size, Image.ANTIALIAS)
    return img


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    tmp = np.zeros(num_classes)
    tmp[labels_dense] = 1
    return tmp


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)
