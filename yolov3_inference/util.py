from __future__ import division

import torch
import numpy as np

def bbox_iou(bbox1, bbox2):
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    if torch.cuda.is_available():
        inter_area = torch.max((inter_rect_x2 - inter_rect_x1 + 1), torch.zeros(inter_rect_x2.shape).cuda()) * \
                     torch.max((inter_rect_y2 - inter_rect_y1 + 1), torch.zeros(inter_rect_y2.shape).cuda())
    else:
        inter_area = torch.max((inter_rect_x2 - inter_rect_x1 + 1), torch.zeros(inter_rect_x2.shape)) * \
                     torch.max((inter_rect_y2 - inter_rect_y1 + 1), torch.zeros(inter_rect_y2.shape))

    bbox1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    bbox2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (bbox1_area + bbox2_area - inter_area)

    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]   #对应到特征图的大小上

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])  # x过sigmoid
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])  # y过sigmoid
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])  # obj过sigmoid

    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1, 1)  # (169, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)  # (169, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    xy_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)  # (1, 169*3, 2)
    prediction[:, :, :2] += xy_offset   #将x,y坐标预测加上相应的cell位置偏置

    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4])*anchors

    prediction[:, :, 5: 5+num_classes] = torch.sigmoid(prediction[:, :, 5: 5+num_classes]) #对class score做sigmoid

    prediction[:, :, 0:4] *= stride

    return prediction  #(batch_size, 13*13*3(每级特征图长宽乘以3), 5+80)


def write_results(prediction, confidence, num_classes, nms=True, nms_conf = 0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    try:
        ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()  #保证有obj大于阈值
    except:
        return 0  #没有obj大于阈值则退出

    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)  #左上角x
    box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)  #左上角y
    box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)  #右上角x
    box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)  #右上角y
    prediction[:, :, :4] = box_a[:, :, :4]  #用左上角和右下角的方式代替预测输出的中心点和宽高

    batch_size = prediction.size(0)
    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]  #每个批次的预测， (1575, 85)

        max_conf, max_conf_score = torch.max(image_pred[:, 5: 5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)  #class最大分数值(1575, 1)
        max_conf_score = max_conf_score.float().unsqueeze(1)   #class最大分数id(1575, 1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1) # (1575, 7)

        non_zero_ind = torch.nonzero(image_pred[:,4])
        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)  #去除obj小于阈值的box
        try:
            img_classes = torch.unique(image_pred_[:, -1])  #提取不重复的类别编号
        except:
            continue

        for cls in img_classes:
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)  #将不是cls类别的置0
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()  #找出class分数不为0的id号
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)  #找出class分数不为0的预测
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]  #找出这类中分数从大到小排列id
            image_pred_class = image_pred_class[conf_sort_index]  #从大到小重新排列预测
            idx = image_pred_class.size(0)

            if nms:
                for i in range(idx):
                    try:
                        # 和后面所有bbox算iou
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
                    except IndexError:  #当有bbox被移除后idx会比原来的idx少，从而会触发indexError，此时跳出循环结束nms
                        break

                    iou_mask = (ious < nms_conf).float().unsqueeze(1)  #找出小于nms阈值的编号
                    image_pred_class[i+1:] *= iou_mask

                    non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_id = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = (batch_id, image_pred_class)
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    return output  #(batch_size_id, x1, y1, x2, y2, obj_conf, class_conf, class_id)