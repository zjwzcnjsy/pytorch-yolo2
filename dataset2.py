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
from utils import image_to_tensor


class ListDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0,
                 batch_size=64, num_workers=4):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train and index % self.batch_size == 0:
            if self.seen < 4000 * self.batch_size:
                width = 13 * 32
                self.shape = (width, width)
            elif self.seen < 8000 * self.batch_size:
                width = (random.randint(0, 3) + 13) * 32
                self.shape = (width, width)
            elif self.seen < 12000 * self.batch_size:
                width = (random.randint(0, 5) + 12) * 32
                self.shape = (width, width)
            elif self.seen < 16000 * self.batch_size:
                width = (random.randint(0, 7) + 11) * 32
                self.shape = (width, width)
            else:  # self.seen < 20000*64:
                width = (random.randint(0, 9) + 10) * 32
                self.shape = (width, width)

        out_w, out_h = self.shape

        if self.train:
            jitter = 0.3
            hue = 0.1
            saturation = 1.5
            exposure = 1.5

            image = cv2.imread(imgpath)
            assert image is not None
            origin_height, origin_width = image.shape[:2]

            if float(out_w) / origin_width < float(out_h) / origin_height:
                new_w = out_w
                new_h = int((origin_height * out_w) / origin_width)
            else:
                new_h = out_h
                new_w = int((origin_width * out_h) / origin_height)

            resized = cv2.resize(image, (new_w, new_h), cv2.INTER_LINEAR)

            dw = int(jitter * resized.shape[1])
            dh = int(jitter * resized.shape[0])

            sized = np.full((out_h + 2 * dh, out_w + 2 * dw, image.shape[2]), 0.5 * 255, dtype=np.float32)
            dx1 = int((out_w - new_w) / 2.)
            dy1 = int((out_h - new_h) / 2.)
            sized[dy1 + dh:dy1 + dh + new_h, dx1 + dw:dx1 + dw + new_w, :] = resized.astype(np.float32)

            dx = random.randint(0, 2*dw)
            dy = random.randint(0, 2*dh)

            sized = sized[dy:dy + out_h, dx:dx + out_w, :]

            dx = (dw - dx + dx1) / out_w
            dy = (dh - dy + dy1) / out_h

            sx = new_w / out_w
            sy = new_h / out_h

            img = random_distort_image(sized, hue, saturation, exposure)

            flip = (random.random() < 0.5)
            if flip:
                img = cv2.flip(img, 1)

            image = img

            labpath = imgpath.replace('images', 'labels') \
                .replace('JPEGImages', 'labels') \
                .replace('.jpg', '.txt') \
                .replace('.png', '.txt')

            label = fill_truth_detection2(labpath, img.shape[1], img.shape[0], flip, -dx, -dy, sx, sy)
            label = torch.from_numpy(label)
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

            tmp = tmp.flatten()
            tsz = tmp.size
            label = np.full((50 * 5,), -1, np.float32)
            if tsz > 50 * 5:
                label = tmp[0:50 * 5]
            elif tsz > 0:
                label[0:tsz] = tmp
            label = torch.from_numpy(label)

            image = sized

        img = image_to_tensor(image / 255.)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return img, label
