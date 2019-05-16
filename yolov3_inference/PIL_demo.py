from __future__ import division
import torch
from darknet import Darknet
from util import write_results
from config import opt
from torch.autograd import Variable
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from skimage import transform as skt

def read_image(img_path, dtype=np.float32, color=True):
    src_img = Image.open(img_path)
    img_ = src_img.copy()
    try:
        if color:
            img_ = img_.convert('RGB')
        else:
            img_ = img_.convert('P')  #模式“P”为8位彩色图像，它的每个像素用8个bit表示，其对应的彩色值是按照调色板查询出来的.
        img_ = np.asarray(img_, dtype=dtype)    #将图片转为array形式
    finally:
        if hasattr(img_, 'close'):  #判断img是否有'close'
            img_.close()

    if img_.ndim == 2:
        return src_img, img_[np.newaxis]
    if img_.ndim == 3:
        return src_img, img_.transpose(2, 0, 1)  #转换为CHW格式


def draw_lable(draw, x1, y1, x2, y2, font, label, color):
    draw.rectangle((x1, y1, x2, y2), fill=color)
    draw.text((x1, y1), label, font=font, fill=(0, 0, 0))


def test_image(img_path):
    src_im, im = read_image(img_path, color=True)
    im = skt.resize(im, (3, opt.resize_img_size, opt.resize_img_size), mode='reflect', anti_aliasing=False)
    im = im / 255.
    im = torch.from_numpy(im)[None].cuda()

    model = Darknet(opt.cfg_path)
    model.load_weights(opt.weightfile)
    model.net_info['height'] = opt.resize_img_size
    model.cuda()
    model.eval()
    pred = model(Variable(im), CUDA=True)
    output = write_results(pred, opt.obj_confidence, opt.num_classes, nms=opt.nms, nms_conf=opt.nms_confidence)
    output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(opt.resize_img_size)) / int(opt.resize_img_size)
    output[:, [1,3]] *= src_im.size[0]
    output[:, [2,4]] *= src_im.size[1]

    label_ids = [id.strip() for id in open(opt.classes_names)]
    draw = ImageDraw.Draw(src_im)
    for i in range(output.shape[0]):
        tl = (output[i, 1], output[i, 2])
        br = (output[i, 3], output[i, 4])
        label = label_ids[int(output[i, 7])]
        color = opt.coco_color[int(output[i, 7])]
        print(color)
        bbox_area = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
        if bbox_area > 128 * 128:
            size = 30
            width = 4
        elif bbox_area > 64 * 64:
            size = 20
            width = 3
        else:
            size = 10
            width = 2
        draw.rectangle((tl[0], tl[1], br[0], br[1]), outline=color, width=width)
        if opt.display_label:
            font = ImageFont.truetype(opt.fontpath, size=size)
            font_size = font.getsize(label)
            if tl[1] > font_size[1]:
                if (tl[0] + font_size[0]) > src_im.size[0]:
                    draw_lable(draw, br[0] - font_size[0], tl[1] - font_size[1], br[0], tl[1], font, label, color)
                else:
                    draw_lable(draw, tl[0], tl[1]-font_size[1], tl[0] + font_size[0], tl[1], font, label, color)
            else:
                if (tl[0] + font_size[0]) > src_im.size[0]:
                    draw_lable(draw, br[0] - font_size[0], tl[1], br[0], tl[1] + font_size[1], font, label, color)
                else:
                    draw_lable(draw, tl[0], tl[1], tl[0] + font_size[0], tl[1] + font_size[1], font, label, color)
    src_im.show()

if __name__ == '__main__':
    test_image(opt.img_path)
