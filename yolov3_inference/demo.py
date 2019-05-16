from __future__ import division
import torch
import cv2
from darknet import Darknet
from util import write_results
from config import opt
from torch.autograd import Variable
import time

def image_transform(srcimg, img_size, cuda=True):
    img = srcimg.copy()
    img = cv2.resize(img, (img_size, img_size))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()  #先把BGR换成RGB，然后把HWC -> CHW
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    if cuda:
        img_ = img_.cuda()
    return img_

def test_image(img_path):
    src_img = cv2.imread(img_path)
    img = image_transform(src_img, opt.resize_img_size, cuda=True)
    model = Darknet(opt.cfg_path)
    model.load_weights(opt.weightfile)
    model.net_info['height'] = opt.resize_img_size
    model.cuda()
    model.eval()
    pred = model(Variable(img), CUDA=True)
    output = write_results(pred, opt.obj_confidence, opt.num_classes, nms=opt.nms, nms_conf=opt.nms_confidence)
    output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(opt.resize_img_size)) / int(opt.resize_img_size)
    output[:, [1,3]] *= src_img.shape[1]
    output[:, [2,4]] *= src_img.shape[0]

    label_ids = [id.strip() for id in open(opt.classes_names)]

    for i in range(output.shape[0]):
        tl = (output[i, 1], output[i, 2])
        br = (output[i, 3], output[i, 4])
        label = label_ids[int(output[i, 7])]
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0]
        color = opt.coco_color[int(output[i, 7])]
        cv2.rectangle(src_img, tl, br,  color, 2)
        if opt.display_label:
            if(tl[1] > 20):
                cv2.rectangle(src_img, tl, (tl[0] + t_size[0], tl[1] - t_size[1] - 5), color, -1)
                cv2.putText(src_img, label, (tl[0], tl[1] - t_size[1] + 10), cv2.FONT_HERSHEY_DUPLEX,
                            0.7, (0,0,0), 1)
            else:
                cv2.rectangle(src_img, tl, (tl[0] + t_size[0], tl[1] + t_size[1] + 11), color, -1)
                cv2.putText(src_img, label, (tl[0], tl[1] + t_size[1] + 4), cv2.FONT_HERSHEY_DUPLEX,
                            0.7, (0,0,0), 1)
    cv2.imshow('src', src_img)
    cv2.waitKey(0)

def test_cam(videopath):
    model = Darknet(opt.cfg_path)
    model.load_weights(opt.weightfile)
    model.net_info['height'] = opt.resize_img_size
    model.cuda()
    model.eval()
    cap = cv2.VideoCapture(videopath)
    start = time.time()
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img = image_transform(frame, opt.resize_img_size, cuda=True)
            pred = model(Variable(img), CUDA=True)
            output = write_results(pred, opt.obj_confidence, opt.num_classes, nms=opt.nms, nms_conf=opt.nms_confidence)

            if type(output) == int:
                frame_num += 1
                print("FPS of the video is {:5.2f}".format(frame_num / (time.time() - start)))
                cv2.imshow("src", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(opt.resize_img_size)) / int(opt.resize_img_size)
            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            label_ids = [id.strip() for id in open(opt.classes_names)]

            for i in range(output.shape[0]):
                tl = (output[i, 1], output[i, 2])
                br = (output[i, 3], output[i, 4])
                label = label_ids[int(output[i, 7])]
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0]
                color = opt.coco_color[int(output[i, 7])]
                cv2.rectangle(frame, tl, br, color, 2)
                if opt.display_label:
                    if (tl[1] > 20):
                        cv2.rectangle(frame, tl, (tl[0] + t_size[0], tl[1] - t_size[1] - 5), color, -1)
                        cv2.putText(frame, label, (tl[0], tl[1] - t_size[1] + 10), cv2.FONT_HERSHEY_DUPLEX,
                                    0.7,(0, 0, 0), 1)
                    else:
                        cv2.rectangle(frame, tl, (tl[0] + t_size[0], tl[1] + t_size[1] + 11), color, -1)
                        cv2.putText(frame, label, (tl[0], tl[1] + t_size[1] + 4), cv2.FONT_HERSHEY_DUPLEX,
                                    0.7, (0, 0, 0), 1)
            cv2.imshow('src', frame)
            frame_num += 1
            print("FPS of the video is {:5.2f}".format(frame_num / (time.time() - start)))
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        else:
            break


if __name__ == '__main__':
    #test_image(opt.img_path)
    test_cam(opt.video_path)