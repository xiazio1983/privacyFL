import hashlib
import json
import sys
import time
from functools import reduce

import numpy as np
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, GT, pair

debug = False


class ShortSig(object):

    @staticmethod
    def param(group):
        # init
        g2 = group.random(G2)
        h = group.random(G2)
        s1, s2 = group.random(ZR), group.random(ZR)
        u, v = h * ~s1, h * ~s2
        gamma = group.random(ZR)
        _lambda = group.random(ZR)
        w = g2 * gamma
        gpk = {'g2': g2, 'h': h, 'u': u, 'v': v, 'w': w, 'lambda': _lambda}
        gmsk = {'s1': s1, 's2': s2}

        return gpk, gmsk, gamma

    @staticmethod
    def gen(group, gpk, gamma):
        # key gen
        x = group.random(ZR)
        A = gpk['g2'] * ~(gamma + x)
        pk = x * gpk['g2']
        return A, x, pk

    @staticmethod
    def agree(sk, pk):
        return sk * pk

    @staticmethod
    def sign(group, gpk, gsk):
        # print(f'raw: {group.hash(M, ZR)}')
        start = time.time()
        alpha, beta = group.random(), group.random()
        A, x, pk = gsk[0], gsk[1], gsk[2]
        T1 = gpk['u'] * alpha
        T2 = gpk['v'] * beta
        T3 = A + (gpk['h'] * (alpha + beta))
        delta1 = x * alpha
        delta2 = x * beta
        r = [group.random() for i in range(5)]
        R1 = gpk['u'] * r[0]
        R2 = gpk['v'] * r[1]
        R3 = (pair(T3, gpk['g2']) ** r[2]) * (pair(gpk['h'], gpk['w'] * (-r[0] - r[1]) + gpk['g2'] * (-r[3] - r[4])))
        R4 = (T1 * r[2]) - (gpk['u'] * r[3])
        R5 = (T2 * r[2]) - (gpk['v'] * r[4])
        timestamp = int(time.time())
        c = group.hash((pk, T1, T2, T3, R1, R2, R3, R4, R5, timestamp), ZR)
        s1, s2 = r[0] + c * alpha, r[1] + c * beta
        s3, s4 = r[2] + c * x, r[3] + c * delta1
        s5 = r[4] + c * delta2
        return {'T1': T1, 'T2': T2, 'T3': T3, 'c': c, 's_alpha': s1, 's_beta': s2, 's_x': s3, 's_delta1': s4,
                's_delta2': s5, 'R3': R3, }, pk, timestamp

    @staticmethod
    def sign2(group, gpk, gsk, data):
        # print(f'raw: {group.hash(M, ZR)}')
        start = time.time()
        alpha, beta = group.random(), group.random()
        A, x, pk = gsk[0], gsk[1], gsk[2]
        T1 = gpk['u'] * alpha
        T2 = gpk['v'] * beta
        T3 = A + (gpk['h'] * (alpha + beta))
        delta1 = x * alpha
        delta2 = x * beta
        r = [group.random() for i in range(5)]
        R1 = gpk['u'] * r[0]
        R2 = gpk['v'] * r[1]
        R3 = (pair(T3, gpk['g2']) ** r[2]) * (pair(gpk['h'], gpk['w'] * (-r[0] - r[1]) + gpk['g2'] * (-r[3] - r[4])))
        R4 = (T1 * r[2]) - (gpk['u'] * r[3])
        R5 = (T2 * r[2]) - (gpk['v'] * r[4])
        timestamp = int(time.time())
        c = group.hash((data, T1, T2, T3, R1, R2, R3, R4, R5, timestamp), ZR)
        s1, s2 = r[0] + c * alpha, r[1] + c * beta
        s3, s4 = r[2] + c * x, r[3] + c * delta1
        s5 = r[4] + c * delta2
        return {'T1': T1, 'T2': T2, 'T3': T3, 'c': c, 's_alpha': s1, 's_beta': s2, 's_x': s3, 's_delta1': s4,
                's_delta2': s5, 'R3': R3, }, data, timestamp

    @staticmethod
    def batch_verify(group, gpk, msgs):
        validSignature = False
        sigmas = []
        for _ in msgs:
            sign, msg, timestamp = _
            sigmas.append(sign)
            sign['delta'] = group.random(ZR)
            r1_ = gpk['u'] * sign['s_alpha'] + sign['T1'] * -sign['c']
            r2_ = gpk['v'] * sign['s_beta'] + sign['T2'] * -sign['c']
            r4_ = sign['T1'] * sign['s_x'] - gpk['u'] * sign['s_delta1']
            r5_ = sign['T2'] * sign['s_x'] - gpk['v'] * sign['s_delta2']
            c_ = group.hash((msg, sign['T1'], sign['T2'], sign['T3'], r1_, r2_, sign['R3'], r4_, r5_, timestamp),
                            ZR)
            if c_ != sign['c']:
                print('False')
                return False

        r3_left = []
        r3_right = []
        r3 = 1
        for sign in sigmas:
            r3_left.append((sign['T3'] * sign['s_x'] - (sign['s_delta1'] + sign['s_delta2']) * gpk['h'] - gpk[
                'g2'] * sign['c']) * sign['delta'])
            r3_right.append((sign['T3'] * sign['c'] - (sign['s_alpha'] + sign['s_beta']) * gpk['h']) * sign['delta'])
            r3 *= (sign['R3'] * sign['delta'])
        r3_ = (pair(reduce(lambda x, y: x + y, r3_left), gpk['g2'])) * (
            pair(reduce(lambda x, y: x + y, r3_right), gpk['w']))
        # r3 = reduce(lambda x, y: (x['R3'] * x['delta']) * (y['R3'] * y['delta']), sigmas)

        if r3_ == r3:
            return True
        else:
            return False


if __name__ == '__main__':
    group = PairingGroup('SS512')

    gpk, gmsk, gamma = ShortSig.param(group)

    users = []
    times = []
    for i in range(10):
        tic = time.time()
        sk, pk, A = ShortSig.gen(group, gpk, gamma)
        users.append((sk, pk, A))
        times.append(int((time.time() - tic) * 1000))
    print(f'KG.gen average time: {sum(times) / len(times)}ms')

    signs = []
    times = []
    for i in users:
        tic = time.time()
        sigma, pki, timestamp = ShortSig.sign(group, gpk, i)
        signs.append((sigma, pki, timestamp))
        times.append(int((time.time() - tic) * 1000))
    print(f'sign average time: {sum(times) / len(times)}ms')

    tic = time.time()
    print(ShortSig.batch_verify(group, gpk, signs))
    print(f'batch verify used: {int((time.time() - tic) * 1000)}ms')
