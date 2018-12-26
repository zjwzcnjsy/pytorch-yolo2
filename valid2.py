from darknet import Darknet
import dataset
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import os
import cv2
import numpy as np
from utils import *
from image import letterbox_image


def image_to_tensor(image):
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.float32
    image = image[:, :, ::-1].transpose((2, 0, 1)).copy()
    image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
    return image


def save_outputs(filename, outputs):
    outputs2 = dict()
    for k, v in outputs.items():
        outputs2[str(k)] = v.cpu().numpy()
    np.savez(filename, **outputs2)


def valid(datacfg, cfgfile, weightfile, outfile):
    cudnn.enabled = True
    cudnn.benchmark = True

    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    name_list = options['names']
    prefix = 'results'
    names = load_class_names(name_list)

    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]

    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    m.cuda()
    m.eval()
    print('shape:', m.width, 'x', m.height)

    fps = []
    if not os.path.exists('results'):
        os.mkdir('results')
    for i in range(m.num_classes):
        buf = '%s/%s%s.txt' % (prefix, outfile, names[i])
        fps.append(open(buf, 'w'))

    conf_thresh = 0.005
    nms_thresh = 0.45
    for batch_idx, valid_file in enumerate(valid_files):
        image = cv2.imread(valid_file)
        assert image is not None
        image2 = letterbox_image(image, m.width, m.height)
        if batch_idx == 0:
            cv2.imwrite('letterbox_image.jpg', image2.astype(np.uint8))
        image_tensor = image_to_tensor(image2)

        data = image_tensor.cuda()
        with torch.no_grad():
            output = m(data)
        # if batch_idx == 0:
        #     outputs[-1] = data
        #     save_outputs('./outputs.npz', outputs)
        batch_boxes = get_region_boxes2(output, image.shape[1], image.shape[0], m.width, m.height, conf_thresh, m.num_classes, m.anchors, m.num_anchors, 1)

        fileId = os.path.basename(valid_file).split('.')[0]
        height, width = image.shape[:2]
        print('[{}/{}]: '.format(batch_idx, len(valid_files)), valid_file, ' ', len(batch_boxes[0]))
        boxes = batch_boxes[0]
        boxes = nms_class(boxes, nms_thresh, m.num_classes)
        for box in boxes:
            x1 = (box[0] - box[2] / 2.0) * width
            y1 = (box[1] - box[3] / 2.0) * height
            x2 = (box[0] + box[2] / 2.0) * width
            y2 = (box[1] + box[3] / 2.0) * height

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 >= width:
                x2 = width - 1
            if y2 >= height:
                y2 = height - 1

            for j in range(m.num_classes):
                prob = box[5 + j]
                if prob >= conf_thresh:
                    fps[j].write('%s %f %f %f %f %f\n' % (fileId, prob, x1, y1, x2, y2))

    for i in range(m.num_classes):
        fps[i].close()


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 4:
        datacfg = sys.argv[1]
        cfgfile = sys.argv[2]
        weightfile = sys.argv[3]
        outfile = 'comp4_det_test_'
        valid(datacfg, cfgfile, weightfile, outfile)
    else:
        print('Usage:')
        print(' python valid2.py datacfg cfgfile weightfile')
