#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import read_truths_args, read_truths
from image import *


def image_to_tensor(image):
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.float32
    image = image[:, :, ::-1].transpose((2, 0, 1)).copy()
    image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
    return image


class ListDataset(Dataset):

    def __init__(self, root, shape=(608, 608), shuffle=True, train=False):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples = len(self.lines)
        self.train = train
        self.shape = shape

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        out_w, out_h = self.shape

        if self.train:
            jitter = 0.3
            hue = 0.1
            saturation = 1.5
            exposure = 1.5

            image = cv2.imread(imgpath)
            assert image is not None
            origin_height, origin_width = image.shape[:2]

            dw = origin_width * jitter
            dh = origin_height * jitter

            pleft = int(random.uniform(-dw, dw))
            pright = int(random.uniform(-dw, dw))
            ptop = int(random.uniform(-dh, dh))
            pbot = int(random.uniform(-dh, dh))

            swidth = origin_width - pleft - pright
            sheight = origin_height - ptop - pbot

            pleft2, pright2, ptop2, pbot2 = map(abs, [pleft, pright, ptop, pbot])

            image2 = cv2.copyMakeBorder(image, ptop2, pbot2, pleft2, pright2, cv2.BORDER_REPLICATE)
            croped = image2[ptop2+ptop:ptop2+ptop+sheight, pleft2+pleft:pleft2+pleft+swidth, :]

            sized, new_w, new_h, dx3, dy3 = letterbox_image(croped, out_w, out_h, return_dxdy=True)

            img = random_distort_image(sized, hue, saturation, exposure)

            flip = (random.random() < 0.5)
            if flip:
                img = cv2.flip(img, 1)

            image = img

            labpath = imgpath.replace('images', 'labels') \
                .replace('JPEGImages', 'labels') \
                .replace('.jpg', '.txt') \
                .replace('.png', '.txt')

            label = np.loadtxt(labpath)
            if label is None:
                label = np.full((5,), -1, dtype=np.float32)
            else:
                label2 = np.full((label.size//5, 5), -1, np.float32)
                bs = np.reshape(label, (-1, 5))
                cc = 0
                for i in range(bs.shape[0]):
                    x1 = bs[i][1] - bs[i][3] / 2
                    y1 = bs[i][2] - bs[i][4] / 2
                    x2 = bs[i][1] + bs[i][3] / 2
                    y2 = bs[i][2] + bs[i][4] / 2

                    x1 = min(swidth, max(0, x1 * origin_width - pleft))
                    y1 = min(sheight, max(0, y1 * origin_height - ptop))
                    x2 = min(swidth, max(0, x2 * origin_width - pleft))
                    y2 = min(sheight, max(0, y2 * origin_height - ptop))

                    x1 = (x1 / swidth * new_w + dx3) / out_w
                    y1 = (y1 / sheight * new_h + dy3) / out_h
                    x2 = (x2 / swidth * new_w + dx3) / out_w
                    y2 = (y2 / sheight * new_h + dy3) / out_h

                    bs[i][1] = (x1 + x2) / 2
                    bs[i][2] = (y1 + y2) / 2
                    bs[i][3] = (x2 - x1)
                    bs[i][4] = (y2 - y1)

                    if flip:
                        bs[i][1] = 0.999 - bs[i][1]

                    if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                        continue
                    label2[cc] = bs[i]
                    cc += 1
                    if cc >= 50:
                        break
                label = label2[:cc].flatten()
        else:
            image = cv2.imread(imgpath)
            assert image is not None
            sized, new_w, new_h, dx, dy = letterbox_image(image, out_w, out_h, return_dxdy=True)

            labpath = imgpath.replace('images', 'labels') \
                .replace('JPEGImages', 'labels') \
                .replace('.jpg', '.txt').replace('.png', '.txt')

            tmp = read_truths_args(labpath, 8.0 / image.shape[1]).astype(np.float32)
            tmp[:, 1:] = (tmp[:, 1:] * np.array([new_w, new_h, new_w, new_h]) + np.array([dx, dy, 0, 0])) / np.array(
                [out_w, out_h, out_w, out_h])

            label = tmp.flatten()
            image = sized

        return dict(image=image, label=label)


class BatchDataCollate:
    def __init__(self, sizes=(416,), processed_batches=0):
        self.sizes = sizes
        self.num_sizes = len(self.sizes)
        self.processed_batches = processed_batches
        self.cur_idx = 0

    def __call__(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""

        error_msg = "batch must contain dicts; found {}"
        assert isinstance(batch[0], dict), error_msg.format(type(batch[0]))
        # print(type(batch), type(batch[0]), batch[0].keys())
        max_label_len = 5
        for dd in batch:
            label = dd['label']
            max_label_len = max(max_label_len, label.size)
        if self.processed_batches % 10 == 0:
            self.cur_idx = random.randint(0, self.num_sizes-1)
        self.processed_batches += 1
        images = [t['image'] for t in batch]
        images = [cv2.resize(t, (self.sizes[self.cur_idx], self.sizes[self.cur_idx])) for t in images]
        images = [image_to_tensor(t.astype(np.float32)) for t in images]
        image_tensor = torch.cat(images, 0)
        lables = np.full((len(batch), max_label_len), -1, dtype=np.float32)
        for i in range(len(batch)):
            label = batch[i]['label']
            lables[i, :label.size] = label
        lables = torch.from_numpy(lables)
        return dict(image=image_tensor, label=lables)
