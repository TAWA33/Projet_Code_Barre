# -*- coding: utf-8 -*-

# Script to test the functions of the Lecture_code_barre package

import numpy as np
import matplotlib.pyplot as plt
from Lecture_code_barre.ean13 import EAN13Codes
import Lecture_code_barre.signature as sgn
import Lecture_code_barre.decodage as decode
from skimage.filters import threshold_otsu

# Initialization
U_MAX = 15
codes = EAN13Codes(U_MAX)


# Test ean13.py: compare a few rows with table A.2 of the PDF
A = codes.families["A"]
B = codes.families["B"]

A_0 = np.array([1, 1, 1, 0, 0, 1, 0])   # BBBNNBN
A_9 = np.array([1, 1, 1, 0, 1, 0, 0])   # BBBNBNN
B_0 = np.array([1, 0, 1, 1, 0, 0, 0])   # BNBBNNN
B_5 = np.array([1, 0, 0, 0, 1, 1, 0])   # BNNNBBN

print("A-0 :", np.array_equal(A[0], A_0))
print("A-9 :", np.array_equal(A[9], A_9))
print("B-0 :", np.array_equal(B[0], B_0))
print("B-5 :", np.array_equal(B[5], B_5))
print("Shape for u = 3 :", codes.get_codes(3)["A"].shape)


# Image and ray (2 clicks)
img = sgn.load_image("./img/beef.jpg")
img = sgn.convert_to_gray(img)

plt.figure(1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title("Grey-level image: click the 2 ends of the ray")
pts = plt.ginput(2, timeout=0)
(x1, y1), (x2, y2) = pts


# First signature
L = np.ceil(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
(x_s, y_s, signature) = sgn.signature_extraction(img, x1, y1, x2, y2, L)

plt.figure(2)
plt.plot(signature)
plt.ylim(0, 255)
plt.title("First signature along the ray")
plt.xlabel("Sample index")
plt.ylabel("Intensity")
plt.savefig('./figures/signature_1.png')

plt.figure(3)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.plot(x_s, y_s, 'r')
plt.plot([x1, x2], [y1, y2], 'rx')
plt.title("Initial ray")
plt.savefig('./figures/rayon_initial.png')

plt.show()


# Otsu threshold
(k, crit) = sgn.otsu_threshold(signature)
print("Otsu threshold (ours) :", k)
print("Otsu threshold (skimage) :", threshold_otsu(signature))

plt.figure(4)
plt.plot(crit)
plt.axvline(k, color='r', linestyle='--', label="k = " + str(k))
plt.title("Otsu criterion")
plt.xlabel("k (gray level)")
plt.ylabel("crit(k)")
plt.legend()
plt.savefig('./figures/otsu_criterion.png')

plt.show()


# Binarization and useful ray
binarized = sgn.binarize_signature(signature, k)

result = sgn.find_useful_ray(binarized, x_s, y_s)
if result is None:
    print("No black sample on this ray")
else:
    (x1_useful, y1_useful, x2_useful, y2_useful, i_first, i_last) = result
    print("First black sample :", i_first, " Last black sample :", i_last)
    print("Useful ray : (", x1_useful, ",", y1_useful, ") --> (", x2_useful, ",", y2_useful, ")")

    plt.figure(5)
    plt.plot(binarized)
    plt.axvline(i_first, color='r', linestyle='--', label="first black sample")
    plt.axvline(i_last, color='g', linestyle='--', label="last black sample")
    plt.ylim(-0.1, 1.1)
    plt.title("Binarized signature (black = 0, white = 1)")
    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig('./figures/signature_binarized.png')

    plt.figure(6)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.plot(x_s, y_s, 'r', label="initial ray")
    plt.plot([x1_useful, x2_useful], [y1_useful, y2_useful], 'g', label="useful ray")
    plt.title("Initial ray and useful ray")
    plt.legend()
    plt.savefig('./figures/useful_ray.png')

plt.show()


# Second signature (along the useful ray, 95u samples)
u = sgn.compute_u(x1_useful, y1_useful, x2_useful, y2_useful)
if u is None:
    print("Barcode too small (useful ray < 95 px)")
else:
    L_2 = 95 * u
    (x_s2, y_s2, signature_2) = sgn.signature_extraction(img, x1_useful, y1_useful, x2_useful, y2_useful, L_2)
    binarized_2 = sgn.binarize_signature(signature_2, k)

    print("u =", u, " L_2 =", L_2)
    print("Length of the second signature :", len(binarized_2))
    print("First and last values :", binarized_2[0], binarized_2[-1])

    plt.figure(7)
    plt.plot(binarized_2)
    for b in [3, 45, 50, 92]:
        plt.axvline(b * u, color='r', linestyle='--')
    plt.ylim(-0.1, 1.1)
    plt.title("Second binarized signature (u = " + str(u) + ")")
    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.savefig('./figures/signature_2_binarized.png')

plt.show()


# Extraction of the partial signature in preparation for decoding
s_p = decode.extract_partial_signature(binarized_2, u)
print(s_p.shape)

# Decoding the barcode
(digits, families, scores) = decode.decode_digits(s_p, codes, u)
print("Digits :", digits)
print("Families :", families)
print("Scores :", scores)

first_digit = decode.decode_first_digit(families, codes.first_digit)
print("first digit =", first_digit)

all_digits = np.insert(digits, 0, first_digit)
control_key = decode.check_control_key(all_digits)
if control_key:
    print("Barcode is valid")
else:
    print("Barcode is invalid")
