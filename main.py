import numpy as np
from utils import img_FFT, img_FFT_inverse
import cv2
import os

result_folder_path = "result/"


def findpower2(num):
    """find the nearest number that is the power of 2"""
    if num & (num-1) == 0:
        return num

    bin_num = bin(num)
    origin_bin_num = str(bin_num)[2:]
    near_power2 = pow(10, len(origin_bin_num))
    near_power2 = "0b" + str(near_power2)
    near_power2 = int(near_power2, base=2)

    return near_power2


def image_padding(img):
    """ padding the image size to power of 2, for fft computation requirement"""
    if len(img.shape) == 2 or len(img.shape) == 3:
        h, w = img.shape[0], img.shape[1]

        h_pad = findpower2(h)-h
        w_pad = findpower2(w)-w

        img = np.pad(img, pad_width=((0, h_pad), (0, w_pad), (0, 0)), mode='constant')

        return img


def image_fft(data_root, img_name):
    """ read, padding, fft, cut to origin size and save """
    if img_name[-3:] != "png" and img_name[-3:] != "tif":
        return 0

    img_path = data_root + img_name

    img_origin = cv2.imread(img_path)
    img = image_padding(img_origin)

    img_fft = img_FFT(img)

    if len(img_origin) == 2:
        img_fft = img_fft[:img_origin.shape[0], :img_origin.shape[1]]
    else:
        img_fft = img_fft[:img_origin.shape[0], :img_origin.shape[1], :]

    img_fft_complex = img_fft.copy()
    # print("img_fft shape")
    # print(img_fft.shape)

    img_fft = np.real(img_fft)

    name, _ = img_name.split(".")
    save_img_name = result_folder_path + name + "_fft.png"
    cv2.imwrite(save_img_name, img_fft)

    return img_fft_complex


def image_dft_inverse(img_fft_complex):
    img_fft = image_padding(img_fft_complex)

    img_origin = img_FFT_inverse(img_fft)
    img_ifft = np.real(img_origin)

    name, _ = img_name.split(".")
    save_img_name = result_folder_path + name + "_inverse.png"

    if len(img_origin) == 2:
        img_ifft = img_ifft[:img_fft_complex.shape[0], :img_fft_complex.shape[1]]
    else:
        img_ifft = img_ifft[:img_fft_complex.shape[0], :img_fft_complex.shape[1], :]

    cv2.imwrite(save_img_name, img_ifft)

    return img_origin


if __name__ == '__main__':

    data_root = "data/"
    result_root = "result/"
    index = 1
    img_list = os.listdir(data_root)
    img_list.sort()
    num = len(img_list)
    for img_name in img_list:
        print("[ %d / %d ] Processing %s" % (index, num, img_name))
        fft_img = image_fft(data_root, img_name)

        if not isinstance(fft_img, int):
            print("fft done")

            image_dft_inverse(fft_img)
            print("inverse fft done")
        index += 1
