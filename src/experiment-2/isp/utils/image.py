import numpy as np
import imageio
import cv2


def read_target_image(path: str, size):
    image = cv2.imread(path)
    if image is None:
        raise Exception(f'Can not read image {path}')
    image = cv2.resize(image, size)
    image = image[:,:,::-1] #bgr -> rgb
    return image.astype(np.float32) / 255


def read_bayer_image(path: str):
    raw = np.asarray(imageio.imread(path))
    if raw is None:
        raise Exception(f'Can not read image {path}')
    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]
    combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    return combined.astype(np.float32) / (4 * 255)