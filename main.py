import numpy as np
from utils import *
import cv2
import os

if __name__ == '__main__':

    # img_path = "data/100.png"
    # fft_img = image_fft(img_path)
    # image_fft_inverse(fft_img, img_path)

    data_root = "data/"
    result_root = "result/"
    index = 1
    img_list = os.listdir(data_root)
    img_list.sort()
    num = len(img_list)
    for img_name in img_list:
        print("[ %d / %d ] Processing %s" % (index, num, img_name))
        img_path = os.path.join(data_root, img_name)

        fft_img = image_fft(img_path)

        if not isinstance(fft_img, int):
            print("fft done")

            image_fft_inverse(fft_img, img_name)
            print("inverse fft done")
        index += 1
