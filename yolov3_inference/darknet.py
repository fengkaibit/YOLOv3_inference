from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from util import predict_transform

def get_test_input(cuda=True):
    img = cv2.imread('data/dog.jpg')
    img = cv2.resize(img, (320, 320))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  #先把BGR换成RGB，在把HWC -> CHW
    img_ = img_[np.newaxis, :,:,:] / 255.0
    if cuda:
        img_ = torch.from_numpy(img_).float().cuda()
    else:
        img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_

def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']  #去除注释行
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode='replicate')
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        global CUDA
        prediction = x
        prediction = predict_transform(prediction, inp_dim, self.anchors, num_classes, CUDA)
        return prediction


def create_modules(blocks):
    net_info = blocks[0]

    module_list = nn.ModuleList()

    index = 0   #为了路由层设置每一层的编号

    prev_filters = 3
    output_filters = []  #记录每一层卷积核的数量,同样是为了路由层

    for x in blocks:
        module = nn.Sequential()
        if (x['type'] == 'net'):
            continue

        if (x['type'] == 'convolutional'):
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module('conv_{0}'.format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)

            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activn)

        elif (x['type'] == 'upsample'):
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=stride, mode='nearest')
            module.add_module('upsample_{0}'.format(index), upsample)

        elif (x['type'] == 'route'):
            x['layers'] = x['layers'].split(',')

            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0

            if start > 0:
                start = start - index

            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module('route_{0}'.format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif (x['type'] == 'shortcut'):
            from_ = int(x['from'])
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)

        elif (x['type'] == 'maxpool'):
            stride = int(x['stride'])
            size = int(x['size'])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = nn.MaxPool2d(size)

            module.add_module('maxpool_{}'.format(index), maxpool)

        elif (x['type'] == 'yolo'):
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)

        else:
            print('Something I donno')
            assert False

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index +=1

    return (net_info, module_list)

class Darknet(nn.Module):


    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def get_blocks(self):
        return self.blocks

    def get_module_list(self):
        return self.module_list

    def forward(self, x, CUDA):
        detections = []
        modules = self.blocks[1:]  #去除第一项net
        outputs = {}  #在字典 outputs 中缓存每个层的输出特征图。键为层的索引，值对应特征图,目的为了路由层的索取

        write = 0
        for i in range(len(modules)):
            module_type = modules[i]['type']
            if module_type == 'convolutional' or module_type == 'upsample' or module_type == 'maxpool':
                x = self.module_list[i](x)  #依次通过module_list
                outputs[i] = x

            elif module_type == 'route':
                layers = modules[i]['layers']
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]  #如果route层只有1个，就把对应的特征图拿过来
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)  #如果route层有2个，就叠加两个层
                outputs[i] = x

            elif module_type == 'shortcut':
                from_ = int(modules[i]['from'])
                x = outputs[i-1] + outputs[i+from_]  #上一层加上from指定层
                outputs[i] = x

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(modules[i]['classes'])

                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                if type(x) == int:
                    continue

                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i-1]  #yolo层为预测层，将特征图编号移到上一层

        try:
            return detections
        except:
            return 0

    def load_weights(self, weightsfile):
        fp = open(weightsfile, 'rb')

        # 第一个 160 比特的权重文件保存了 5 个 int32 值，它们构成了文件的标头。
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. IMages seen
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']

            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0

                conv = model[0]


                #加载bn或者conv_bias
                if batch_normalize:

                    bn = model[1]
                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.data.copy_(bn_running_mean)
                    bn.running_var.data.copy_(bn_running_var)


                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                #加载conv_weight
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

if __name__ == '__main__':
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights('yolov3.weights')
    model.cuda()
    model.eval()
    inp = get_test_input()
    pred = model(inp, CUDA=True)
    from util import write_results
    output = write_results(pred, 0.5, 80, nms=True, nms_conf=0.5)
    print(output)

