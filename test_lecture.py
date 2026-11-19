# -*- coding: utf-8 -*-

# test_lecture.py
# Scratch script to test the functions of the Lecture_code_barre package
# Run from the root folder (Projet_Code_Barre)


import numpy as np
from Lecture_code_barre.ean13 import A, B, C, dict_stretched_u


# Test ean13.py: compare a few rows with table A.2 of the PDF

A_0 = np.array([1, 1, 1, 0, 0, 1, 0])   # A, digit 0: BBBNNBN
A_9 = np.array([1, 1, 1, 0, 1, 0, 0])   # A, digit 9: BBBNBNN
B_0 = np.array([1, 0, 1, 1, 0, 0, 0])   # B, digit 0: BNBBNNN
B_5 = np.array([1, 0, 0, 0, 1, 1, 0])   # B, digit 5: BNNNBBN

print("A-0 :", np.array_equal(A[0], A_0))
print("A-9 :", np.array_equal(A[9], A_9))
print("B-0 :", np.array_equal(B[0], B_0))
print("B-5 :", np.array_equal(B[5], B_5))

# Size of the stretched codes for u = 3 (should be (10, 21))
print("Shape for u = 3 :", dict_stretched_u[3]["A"].shape)

