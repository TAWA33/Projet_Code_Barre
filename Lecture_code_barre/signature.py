# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image, ImageOps


def load_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = np.array(img)
    return img


# Convert any image to grey levels (2D array of floats in 0-255)
def convert_to_gray(img):
    if img.dtype == np.uint8:
        img = img.astype(float)
    else:
        if img.max() <= 1:
            img = img * 255
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = img[:, :, :3]
        Y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return Y
    raise ValueError("convert_to_grey: unexpected image shape " + str(img.shape))



def signature_extraction(img, x1, y1, x2, y2, L):
    x_s = np.linspace(x1, x2, num=int(L))
    y_s = np.linspace(y1, y2, num=int(L))
    signature = map_coordinates(img, [y_s, x_s], order=1)
    return (x_s, y_s, signature)




def otsu_threshold(signature):
    (hist, bin_edges) = np.histogram(signature, bins=256, range=(0, 256))
    w = np.cumsum(hist) / np.sum(hist)
    i_hist = np.arange(256) * hist 
    mue = np.cumsum(i_hist) / np.sum(hist)
    mue_N = np.sum(i_hist) / np.sum(hist)
    crit = w * (mue_N - mue)**2 + (1 - w) * mue**2
    k = np.argmax(crit)
    return (k, crit)




def binarize_signature(signature, k):
    binarized = signature > k
    return binarized.astype(int)



# Scale a signature to 0-1 using the mean of the black samples and the mean of the white samples
# (separated by the Otsu threshold k), then cap the values to 0-1
# def normalize_signature_means(signature, k):
#     mean_black = np.mean(signature[signature <= k])
#     mean_white = np.mean(signature[signature > k])
#     normalized = (signature - mean_black) / (mean_white - mean_black)
#     normalized = np.clip(normalized, 0, 1)
#     return normalized



def find_useful_ray(binarized, x_s, y_s):
    index = np.where(binarized == 0)[0]
    if len(index) == 0:
        return None
    x1_useful, y1_useful = x_s[index[0]], y_s[index[0]]
    x2_useful, y2_useful = x_s[index[-1]], y_s[index[-1]]
    return (x1_useful, y1_useful, x2_useful, y2_useful, index[0], index[-1])




def compute_u(x1_useful, y1_useful, x2_useful, y2_useful):
    L_useful = np.sqrt((x2_useful - x1_useful)**2 + (y2_useful - y1_useful)**2)
    if L_useful < 95:
        return None
    u = np.ceil(L_useful / 95)
    return int(u)

