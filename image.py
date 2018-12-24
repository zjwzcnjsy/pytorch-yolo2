#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image
import numpy as np
import cv2


def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out


def distort_image(im, hue, sat, val):
    np_im = np.asarray(im).astype(np.float32)
    np_im = cv2.cvtColor(np_im, cv2.COLOR_RGB2HSV)
    np_im[:, :, 1] *= sat
    np_im[:, :, 2] *= val

    np_im[:, :, 0] += hue * 255
    np_im[:, :, 0][np_im[:, :, 0] > 255] -= 255
    np_im[:, :, 0][np_im[:, :, 0] < 0] += 255

    im = cv2.cvtColor(np_im, cv2.COLOR_HSV2RGB)
    im[im > 255] = 255
    im[im < 0] = 0
    return im


def distort_image2(im, hue, sat, val):
    assert isinstance(im, np.ndarray)
    np_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    np_im[:, :, 1] *= sat
    np_im[:, :, 2] *= val

    np_im[:, :, 0] += hue * 255
    np_im[:, :, 0][np_im[:, :, 0] > 255] -= 255
    np_im[:, :, 0][np_im[:, :, 0] < 0] += 255

    im = cv2.cvtColor(np_im, cv2.COLOR_HSV2BGR)
    im[im > 255] = 255
    im[im < 0] = 0
    return im


def rand_scale(s):
    scale = random.uniform(1, s)
    if random.random() < 0.5:
        return scale
    return 1. / scale


def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res


def random_distort_image2(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image2(im, dhue, dsat, dexp)
    return res


def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height
    ow = img.width

    dw = int(ow * jitter)
    dh = int(oh * jitter)

    pleft = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    pbot = random.randint(-dh, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth) / ow
    sy = float(sheight) / oh

    flip = (random.random() < 0.5)
    cropped = img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft) / ow) / sx
    dy = (float(ptop) / oh) / sy

    sized = cropped.resize(shape, Image.BILINEAR)

    if flip:
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)

    return img, flip, dx, dy, sx, sy


def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes, 5), dtype=np.float32)
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3] / 2
            y1 = bs[i][2] - bs[i][4] / 2
            x2 = bs[i][1] + bs[i][3] / 2
            y2 = bs[i][2] + bs[i][4] / 2

            x1 = min(0.999, max(0, x1 * sx - dx))
            y1 = min(0.999, max(0, y1 * sy - dy))
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))

            bs[i][1] = (x1 + x2) / 2
            bs[i][2] = (y1 + y2) / 2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] = 0.999 - bs[i][1]

            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label


def fill_truth_detection2(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.full((max_boxes, 5), -1, dtype=np.float32)
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3] / 2
            y1 = bs[i][2] - bs[i][4] / 2
            x2 = bs[i][1] + bs[i][3] / 2
            y2 = bs[i][2] + bs[i][4] / 2

            x1 = min(0.999, max(0, x1 * sx - dx))
            y1 = min(0.999, max(0, y1 * sy - dy))
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))

            bs[i][1] = (x1 + x2) / 2
            bs[i][2] = (y1 + y2) / 2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] = 0.999 - bs[i][1]

            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label


def fill_truth_detection2(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.full((max_boxes, 5), -1, dtype=np.float32)
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3] / 2
            y1 = bs[i][2] - bs[i][4] / 2
            x2 = bs[i][1] + bs[i][3] / 2
            y2 = bs[i][2] + bs[i][4] / 2

            x1 = min(0.999, max(0, x1 * sx - dx))
            y1 = min(0.999, max(0, y1 * sy - dy))
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))

            bs[i][1] = (x1 + x2) / 2
            bs[i][2] = (y1 + y2) / 2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] = 0.999 - bs[i][1]

            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label


def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
    labpath = imgpath.replace('images', 'labels')\
                    .replace('JPEGImages', 'labels')\
                    .replace('.jpg', '.txt')\
                    .replace('.png', '.txt')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    img, flip, dx, dy, sx, sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labpath, img.shape[1], img.shape[0], flip, dx, dy, 1. / sx, 1. / sy)
    return img, label


def letterbox_image(image, w, h, return_dxdy=False):
    assert len(image.shape) == 3
    assert image.shape[-1] == 3
    new_w = 0
    new_h = 0

    if float(w) / image.shape[1] < float(h) / image.shape[0]:
        new_w = w
        new_h = int((image.shape[0] * w) / image.shape[1])
    else:
        new_h = h
        new_w = int((image.shape[1] * h) / image.shape[0])

    resized = cv2.resize(image, (new_w, new_h), cv2.INTER_LINEAR)
    boxed = np.full((h, w, image.shape[2]), 0.5*255, dtype=np.float32)
    dx = int((w - new_w) / 2.)
    dy = int((h - new_h) / 2.)
    boxed[dy:dy+new_h, dx:dx+new_w, :] = resized.astype(np.float32)
    if return_dxdy:
        return boxed, new_w, new_h, dx, dy
    return boxed
