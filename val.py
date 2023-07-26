import torch
import torchvision
import torch.optim
from model import GACA as model
import numpy as np
import torch.nn as nn
from PIL import Image
import glob
import time, os, cv2


def att(channal):
    cv2.normalize(channal, channal, 0, 255, cv2.NORM_MINMAX)
    M = np.ones(channal.shape, np.uint8) * 255
    img_new = cv2.subtract(M, channal)
    return img_new


def process(img):
    # img = np.array(img)
    b, g, r = cv2.split(img)
    b = att(b)
    g = att(g)
    r = att(r)
    new_image = cv2.merge([r, g, b])
    return new_image


def dehaze_image(image_hazy_path):
    # --------------------------------------------------
    img_hazy = np.array(Image.open(image_hazy_path).convert('RGB'))


    if img_hazy.shape[0] > 2000:
        index = int(img_hazy.shape[1] / 2), int(img_hazy.shape[0] / 2)
        img_hazy = cv2.resize(img_hazy, index)
    if img_hazy.shape[0] % 16 != 0 or img_hazy.shape[1] % 16 != 0:
        i = img_hazy.shape[0]
        j = img_hazy.shape[1]
        if img_hazy.shape[0] % 16 != 0:
            while i % 16 != 0:
                i -= 1
        if img_hazy.shape[1] % 16 != 0:
            while j % 16 != 0:
                j -= 1
        img_hazy = cv2.resize(img_hazy, (j, i))

    img_hazy = (np.asarray(img_hazy) / 255.0)
    img_hazy = torch.from_numpy(img_hazy).float().permute(2, 0, 1).cuda().unsqueeze(0)


    with torch.no_grad():
        clean_image,_= net(img_hazy)

    index = image_hazy_path.split('\\')[-1]

    torchvision.utils.save_image(clean_image, "results/%s" % (index))

if __name__ == '__main__':


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    pth_path = "weights/best.pth"  #####
    net = model.Gradi_ContrativeNet().cuda()
    net.load_state_dict(torch.load(pth_path))


    hazy_list= glob.glob("./test/*")####
    print('image num:', len(hazy_list))
    for Id in range(len(hazy_list)):
        dehaze_image(hazy_list[Id])
        print(hazy_list[Id], "done!")
