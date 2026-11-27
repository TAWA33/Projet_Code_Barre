import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import ListedColormap
from math import ceil, floor, sqrt
import skimage as sk
from scipy import signal, ndimage, linalg
from scipy.interpolate import RectBivariateSpline
import cv2

import time

def plot_channels(channels,titles=None,res_factor=1):
    n=len(channels)
    if titles==None:
        titles=[str(i) for i in range(n)]
    h,w=ceil(n/2),2
    print((h,w))
    plt.figure(figsize=(4*res_factor*w, 3*res_factor*h))
    for i in range(n):
        plt.subplot(h,w,i+1)
        plt.imshow(channels[i], cmap='gray')
        plt.title(titles[i])
    
    plt.tight_layout()
    plt.show()
    return

def div2_img(img, sigma_noise):
    # Low pass Gaussian Filter
    print(img.shape)
    sigma_x, sigma_y = sigma_noise
    x = range(floor(-3*sigma_x), floor(3*sigma_x+1))
    y = range(floor(-3*sigma_y), floor(3*sigma_y+1))
    X, Y = np.meshgrid(x, y)
    H = np.exp(-(X**2/(2*sigma_x**2) + Y**2/(2*sigma_y**2)))
    H /= H.sum()
    img_filt = signal.convolve2d(img, H, mode='same')[::2, ::2]
    # Decimation
    print(img_filt.shape)

    return img_filt

def build_pyramid(Y, min_size=300, sigma_noise=(0.5, 0.5)):
    pyr = [Y]
    while min(pyr[-1].shape[:2]) > min_size:
        pyr.append(div2_img(pyr[-1], sigma_noise))
    return pyr

def G_2D(sigma):
    x = range(floor(-3*sigma), floor(3*sigma+1))
    X, Y = np.meshgrid(x, x)
    return np.exp(-1/2*(X**2/(sigma**2)+(Y**2/sigma**2)))

def G_x_prime(sigma):  # Derivate (x) of a G
    P = range(floor(-3*sigma), floor(3*sigma+1))
    X, Y = np.meshgrid(P, P)
    return (-X/(2*np.pi*sigma**4)*np.exp(-(X**2+Y**2)/(2*sigma**2)))


def G_y_prime(sigma):  # Derivate (y) of a G
    P = range(floor(-3*sigma), floor(3*sigma+1))
    X, Y = np.meshgrid(P, P)
    return (-Y/(2*np.pi*sigma**4)*np.exp(-(X**2+Y**2)/(2*sigma**2)))

def compute_gradI(Y, sigma_g):
    # Applying Gaussian mask
    gauss_x_prime = G_x_prime(sigma_g)
    gauss_y_prime = G_y_prime(sigma_g)

    I_x = signal.fftconvolve(Y, gauss_x_prime, mode='same')
    I_y = signal.fftconvolve(Y, gauss_y_prime, mode='same')

    #plot_channels([I_x, I_y], ["I_x", "I_y"])

    # Normalization
    norm = np.sqrt(I_x**2 + I_y**2)
    eps = 1e-8
    I_x_norm = I_x / (norm + eps)
    I_y_norm = I_y / (norm + eps)
    return (I_x_norm, I_y_norm)
    

def compute_T(I_x_norm, I_y_norm, sigma_t):
    gauss_2d = G_2D(sigma_t)
    
    Txx = signal.fftconvolve(I_x_norm * I_x_norm, gauss_2d, mode='same')
    Tyy = signal.fftconvolve(I_y_norm * I_y_norm, gauss_2d, mode='same')
    Txy = signal.fftconvolve(I_x_norm * I_y_norm, gauss_2d, mode='same')
    return (Txx, Txy, Tyy)

def consistency_D(Txx, Tyy, Txy, eps=1e-8):
    return np.sqrt((Txx - Tyy)**2 + 4 * Txy**2) / (Txx + Tyy + eps)

def compute_D_bin(Y, sigma_g, sigma_t, seuil_D):
    I_x_norm, I_y_norm = compute_gradI(Y, sigma_g)   

    Txx, Txy, Tyy = compute_T(I_x_norm, I_y_norm, sigma_t)

    D = consistency_D(Txx, Tyy, Txy)
    D_bin = D > seuil_D

    #plot_channels([Y, D, D_bin], ["Y", "D", f"D > {seuil_D}"])
    return D_bin

def compute_valid_regions(D_bin, min_compacity, max_compacity):
    L = sk.measure.label(D_bin)
    regions = sk.measure.regionprops(L)
    valid_regions = []
    areas = []
    for r in regions:
        areas.append(r.area)
    mean_area = np.mean(areas)
    std_area = np.std(areas)

    for r in regions:
        if r.perimeter > 0:
            compacity = (4 * np.pi * r.area) / (r.perimeter ** 2)
            if (compacity < max_compacity) & (compacity > min_compacity):
                if r.area > mean_area : #+ std_area:
                    valid_regions.append(r)

    return valid_regions


def plot_valid_regions_patches(img, valid_regions):
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    ax = plt.gca()

    for r in valid_regions:
        min_row, min_col, max_row, max_col = r.bbox
        
        rect = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        y_center, x_center = r.centroid
        plt.plot(x_center, y_center, 'b+', markersize=12, markeredgewidth=2)

    plt.title("Detected regions")
    plt.show()

# Parameters
sigma_noise = (0.5, 0.5) # Low pass Gaussian Filter (Reduction dimension image)
sigma_g = 3 # Gaussian standard deviation (for G_x_prime / G_y_prime). Approx u
sigma_t = 10 * sigma_g # Tensor standard deviation.
seuil_D = 0.6
min_compacity = 0.5
max_compacity = 0.8 # Circle=1 / Square=0.78

start_time = time.time()

# Loading Image
img_path = "./img/can.jpg"
I = plt.imread(img_path)

# Transform RGB -> YCbCr
I_YCbCr = sk.color.rgb2ycbcr(I)
Y = I_YCbCr[:, :, 0]
Cb = I_YCbCr[:, :, 1]
Cr = I_YCbCr[:, :, 2]

# Multi-scale Approach
# min_shape = (700, 700)
# while Y.shape > min_shape:
#     Y = div2_img(Y, sigma_noise)

D_bin = compute_D_bin(Y, sigma_g, sigma_t, seuil_D)
valid_regions = compute_valid_regions(D_bin, min_compacity, max_compacity)

print("--- %s seconds ---" % (time.time() - start_time))

plot_valid_regions_patches(I, valid_regions)

