import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from math import ceil, floor, sqrt
import skimage as sk
from scipy import signal, ndimage
from scipy.interpolate import RectBivariateSpline

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

# Parameters
sigma_g = 1 # Gaussian standard deviation (for G_x_prime / G_y_prime)
sigma_t = 1 # Tensor standard deviation

# Loading Image
img_path = "./img/homographie.jpg"
I = plt.imread(img_path)

# Transform RGB -> YCbCr
I_YCbCr = sk.color.rgb2ycbcr(I)
Y = I_YCbCr[:, :, 0]
Cb = I_YCbCr[:, :, 1]
Cr = I_YCbCr[:, :, 2]


# Applying Gaussian mask
gauss_x_prime = G_x_prime(sigma_g)
gauss_y_prime = G_y_prime(sigma_g)
h, w, c = np.shape(I_YCbCr)
x = np.linspace(0, h, h)
y = np.linspace(0, w, w)
X, Y = np.meshgrid(y, x)

I_x = signal.fftconvolve(Y, gauss_x_prime, mode='same')
I_y = signal.fftconvolve(Y, gauss_y_prime, mode='same')

plot_channels([I_x, I_y])

gauss_2d = G_2D(sigma_t)

Txx = signal.fftconvolve(I_x * I_x, gauss_2d, mode='same')
Tyy = signal.fftconvolve(I_y * I_y, gauss_2d, mode='same')
Txy = signal.fftconvolve(I_x * I_y, gauss_2d, mode='same')
T = np.block([[Txx, Txy], [Txy, Tyy]])
