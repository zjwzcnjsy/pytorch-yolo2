from __future__ import print_function
import sys
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms
import torchvision
import visdom
import cv2
import dataset2
from utils import *
from cfg import parse_cfg
from darknet2 import Darknet
from models.tiny_yolo import TinyYoloNet


def parse():
    parser = argparse.ArgumentParser(description='yolo train')
    parser.add_argument('--data-cfg', default='cfg/voc.data',
                        help='voc data config file')
    parser.add_argument('--cfg-file', default='cfg/yolo-voc.cfg',
                        help='yolo cfg file')
    parser.add_argument('--pre-trained-file', default='darknet19_448.conv.23',
                        help='pre trained weight file')
    parser.add_argument('--evaluation', action='store_true',
                        help='do evaluate only')
    args = parser.parse_args()
    return args


def adjust_learning_rate(learning_rate, steps, scales, batch_size, optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / batch_size
    return lr


def draw_boxes(image, boxes, names):
    for cls_id, cx, cy, w, h in boxes:
        x1 = cx - w / 2.
        y1 = cy - h / 2.
        x2 = cx + w / 2.
        y2 = cy + h / 2.
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, '{}'.format(names[cls_id]), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def visdom_images(outputs, targets, names):
    images = (outputs.numpy().copy() * 255).astype(np.uint8)
    batch_size, _, height, width = images.shape
    images = np.ascontiguousarray(images.transpose((0, 2, 3, 1)))
    targets = targets.numpy().reshape(batch_size, -1, 5)
    for i in range(batch_size):
        boxes = []
        for cls_id, cx, cy, w, h in targets[i]:
            if cls_id == -1:
                break
            boxes.append((cls_id, cx * width, cy * height, w * width, h*height))
        images[i] = draw_boxes(images[i], boxes, names)
    grid_images = torchvision.utils.make_grid(torch.from_numpy(images.transpose((0, 3, 1, 2)).astype(np.float32)),
                                              nrow=4)
    return grid_images


if __name__ == '__main__':
    args = parse()
    # Training settings
    datacfg = args.data_cfg
    cfgfile = args.cfg_file
    weightfile = args.pre_trained_file

    data_options = read_data_cfg(datacfg)
    net_options = parse_cfg(cfgfile)[0]

    trainlist = data_options['train']
    testlist = data_options['valid']
    backupdir = data_options['backup']
    nsamples = file_lines(trainlist)
    gpus = data_options['gpus']  # e.g. 0,1,2,3
    ngpus = len(gpus.split(','))
    num_workers = int(data_options['num_workers'])

    names = data_options['names']
    with open(names, mode='r') as fin:
        lines = fin.readlines()
        names = [line.strip() for line in lines if line.strip() != '']
    sizes = [int(t.strip()) for t in data_options['sizes'].split(',')]

    batch_size = int(net_options['batch'])
    max_batches = int(net_options['max_batches'])
    learning_rate = float(net_options['learning_rate'])
    momentum = float(net_options['momentum'])
    decay = float(net_options['decay'])
    steps = [float(step) for step in net_options['steps'].split(',')]
    scales = [float(scale) for scale in net_options['scales'].split(',')]

    # Train parameters
    max_epochs = int(1. * max_batches * batch_size / nsamples + 1)
    use_cuda = True
    eps = 1e-5
    save_interval = 5  # epoches
    dot_interval = 70  # batches

    # Test parameters
    conf_thresh = 0.25
    nms_thresh = 0.4
    iou_thresh = 0.5

    if not os.path.exists(backupdir):
        os.mkdir(backupdir)

    visdom_dir = os.path.join(backupdir, 'visdom')

    if not os.path.exists(visdom_dir):
        os.mkdir(visdom_dir)
    vis = visdom.Visdom(env='yolov2')

    ###############
    seed = int(time.time())
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    model = Darknet(cfgfile)
    region_loss = model.loss
    model.loss = None

    model.load_weights(weightfile)
    model.print_network()

    region_loss.seen = model.seen
    processed_batches = model.seen / batch_size

    init_width = model.width
    init_height = model.height
    init_epoch = int(model.seen / nsamples)

    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    test_loader = torch.utils.data.DataLoader(
        dataset2.ListDataset(testlist, shape=(sizes[-1], sizes[-1]),
                             shuffle=False, train=False),
        collate_fn=dataset2.BatchDataCollate(sizes=sizes),
        batch_size=batch_size, shuffle=False, **kwargs)

    if use_cuda:
        cudnn.enabled = True
        cudnn.benchmark = True
        if ngpus > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay * batch_size}]
    optimizer = optim.SGD(model.parameters(), lr=learning_rate / batch_size, momentum=momentum, dampening=0,
                          weight_decay=decay * batch_size)


    def train(epoch):
        global processed_batches
        t0 = time.time()
        if ngpus > 1:
            cur_model = model.module
        else:
            cur_model = model
        train_loader = torch.utils.data.DataLoader(
            dataset2.ListDataset(trainlist, shape=(sizes[-1], sizes[-1]),
                                 shuffle=True,
                                 train=True),
            collate_fn=dataset2.BatchDataCollate(sizes=sizes, processed_batches=processed_batches),
            batch_size=batch_size, shuffle=False, **kwargs)

        lr = adjust_learning_rate(learning_rate, steps, scales, batch_size, optimizer, processed_batches)
        logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
        model.train()
        t1 = time.time()
        avg_time = torch.zeros(9)
        for batch_idx, loaded_data in enumerate(train_loader):
            data, target = loaded_data['image'], loaded_data['label']
            t2 = time.time()
            adjust_learning_rate(learning_rate, steps, scales, batch_size, optimizer, processed_batches)
            processed_batches = processed_batches + 1
            # if (batch_idx+1) % dot_interval == 0:
            #    sys.stdout.write('.')
            if batch_idx % 100 == 0:
                vis.image(visdom_images(data, target, names), win='train')
                print('plot images...')

            if use_cuda:
                data = data.cuda()
                # target= target.cuda()
            t3 = time.time()
            t4 = time.time()
            optimizer.zero_grad()
            t5 = time.time()
            output = model(data)
            t6 = time.time()
            region_loss.seen = region_loss.seen + data.data.size(0)
            loss = region_loss(output, target)
            t7 = time.time()
            loss.backward()
            t8 = time.time()
            optimizer.step()
            t9 = time.time()
            if False and batch_idx > 1:
                avg_time[0] = avg_time[0] + (t2 - t1)
                avg_time[1] = avg_time[1] + (t3 - t2)
                avg_time[2] = avg_time[2] + (t4 - t3)
                avg_time[3] = avg_time[3] + (t5 - t4)
                avg_time[4] = avg_time[4] + (t6 - t5)
                avg_time[5] = avg_time[5] + (t7 - t6)
                avg_time[6] = avg_time[6] + (t8 - t7)
                avg_time[7] = avg_time[7] + (t9 - t8)
                avg_time[8] = avg_time[8] + (t9 - t1)
                print('-------------------------------')
                print('       load data : %f' % (avg_time[0] / (batch_idx)))
                print('     cpu to cuda : %f' % (avg_time[1] / (batch_idx)))
                print('cuda to variable : %f' % (avg_time[2] / (batch_idx)))
                print('       zero_grad : %f' % (avg_time[3] / (batch_idx)))
                print(' forward feature : %f' % (avg_time[4] / (batch_idx)))
                print('    forward loss : %f' % (avg_time[5] / (batch_idx)))
                print('        backward : %f' % (avg_time[6] / (batch_idx)))
                print('            step : %f' % (avg_time[7] / (batch_idx)))
                print('           total : %f' % (avg_time[8] / (batch_idx)))
            t1 = time.time()
        print('')
        t1 = time.time()
        logging('training with %f samples/s' % (len(train_loader.dataset) / (t1 - t0)))
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == max_epochs:
            logging('save weights to %s/%06d.weights' % (backupdir, epoch + 1))
            cur_model.seen = (epoch + 1) * len(train_loader.dataset)
            cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch + 1))


    def test(epoch):
        def truths_length(truths):
            for i in range(truths.size(0)):
                if truths[i][0] == -1:
                    return i
            return truths.size(0)

        model.eval()
        if ngpus > 1:
            cur_model = model.module
        else:
            cur_model = model
        num_classes = cur_model.num_classes
        anchors = cur_model.anchors
        num_anchors = cur_model.num_anchors
        total = 0.0
        proposals = 0.0
        correct = 0.0

        with torch.no_grad():
            for batch_idx, loaded_data in enumerate(test_loader):
                data, target = loaded_data['image'], loaded_data['label']
                if use_cuda:
                    data = data.cuda()
                output = model(data)
                all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
                for i in range(output.size(0)):
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)
                    truths = target[i].view(-1, 5)
                    num_gts = truths_length(truths)

                    total = total + num_gts

                    for j in range(len(boxes)):
                        if boxes[j][4] > conf_thresh:
                            proposals = proposals + 1

                    for j in range(num_gts):
                        box_gt = [truths[j, 1].item(), truths[j, 2].item(), truths[j, 3].item(), truths[j, 4].item(),
                                  1.0, 1.0, truths[j, 0].item()]
                        best_iou = 0
                        best_j = -1
                        for k in range(len(boxes)):
                            iou = bbox_iou(box_gt, boxes[k], x1y1x2y2=False)
                            if iou > best_iou:
                                best_j = k
                                best_iou = iou
                        if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                            correct = correct + 1

        precision = 1.0 * correct / (proposals + eps)
        recall = 1.0 * correct / (total + eps)
        fscore = 2.0 * precision * recall / (precision + recall + eps)
        logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))


    if args.evaluation:
        logging('evaluating ...')
        test(0)
    else:
        for epoch in range(init_epoch, max_epochs):
            train(epoch)
            test(epoch)
