import os
import os.path
from PIL import Image
import sys
sys.path.append('.')
from darknet import Darknet
from utils import *
import cv2
import numpy as np

def save_boxes(img, boxes, savename):
    fp = open(savename, 'w')
    filename = os.path.basename(savename)
    filename = os.path.splitext(filename)[0]
    fp.write('%s\n' % filename)
    fp.write('%d\n' % len(boxes))
    width = img.shape[1]
    height = img.shape[0]
    for box in boxes:
        x1 = round((box[0] - box[2]/2.0) * width)
        y1 = round((box[1] - box[3]/2.0) * height)
        x2 = round((box[0] + box[2]/2.0) * width)
        y2 = round((box[1] + box[3]/2.0) * height)
        w = x2 - x1
        h = y2 - y1
        conf = box[4]
        fp.write('%d %d %d %d %f\n' % (x1, y1, w, h, conf))
    fp.close()

def pad_wh(img):
    wid = img.shape[1]
    hei = img.shape[0]
    siz = max(wid, hei)
    new_img = np.zeros((siz, siz, 3), np.uint8)
    new_img[:hei, :wid, :] = img
    return new_img

def eval_widerface(cfgfile, weightfile, valdir, savedir):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    use_cuda = 1
    if use_cuda:
        m.cuda()

    scale_size = 16
    class_names = load_class_names('data/names')
    for parent,dirnames,filenames in os.walk(valdir):
        if parent != valdir:
            targetdir = os.path.join(savedir, os.path.basename(parent))
            if not os.path.isdir(targetdir):
                os.mkdir(targetdir)
            for filename in filenames:
                imgfile = os.path.join(parent,filename)
                img = cv2.imread(imgfile)
                orig_wid = img.shape[1]
                orig_hei = img.shape[0]
                img = pad_wh(img)
                img1 = cv2.resize(img, (512, 512))
                img2 = cv2.resize(img, (1024, 1024))
                img3 = cv2.resize(img, (2048, 2048))
                boxes1 = do_detect(m, img1, 0.05, 0.4, use_cuda)
                boxes2 = do_detect(m, img2, 0.05, 0.4, use_cuda)
                boxes3 = do_detect(m, img3, 0.05, 0.4, use_cuda)
                boxes = boxes1 + boxes2 + boxes3
                boxes = nms(boxes, 0.1)
                if True:
                    savename = os.path.join(targetdir, filename)
                    print('save to %s' % savename)
                    plot_boxes_cv2(img, boxes, savename)
                if True:
                    savename = os.path.join(targetdir, os.path.splitext(filename)[0]+".txt")
                    print('save to %s' % savename)
                    save_boxes(img, boxes, savename)

if __name__ == '__main__':
    #eval_widerface('resnet50_test.cfg', 'resnet50_98000.weights', 'widerface/WIDER_val/images/', 'widerface/wider_val_pred/')
    #eval_widerface('resnet50_test.cfg', 'resnet50_148000.weights', 'widerface/WIDER_val/images/', 'widerface/wider_val_pred/')
    eval_widerface('wider4_results/wider4.cfg', 'wider4_results/backup/000050.weights', '/home/xiaohang/wider_face/WIDER_val/images/', 'wider_val_pred/')

