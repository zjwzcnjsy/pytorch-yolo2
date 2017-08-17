import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet

def detect_test_data(datacfg, cfgfile, weightfile):
    data_options = read_data_cfg(datacfg)
    testlist = data_options['valid']
    namesfile = data_options['names']

    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    class_names = load_class_names(namesfile)
    lines = open(testlist).readlines()
    ind = 0
    for line in lines:
        imgpath = line.strip()
        #labpath = imgpath.replace('.jpg', '.txt').replace('images', 'labels')
        img = Image.open(imgpath).convert('RGB')
        sized = img.resize((m.width, m.height))
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        savename = 'plot/%06d.jpg' % ind
        plot_boxes(img, boxes, savename, class_names)
        ind = ind + 1

if __name__ == '__main__':
    if len(sys.argv) == 4:
        datacfg = sys.argv[1]
        cfgfile = sys.argv[2]
        weightfile = sys.argv[3]
        detect_test_data(datacfg, cfgfile, weightfile)
    else:
        print('Usage: ')
        print('  python plot_results.py datacfg cfgfile weightfile')
