# -*- coding: utf-8 -*-

# ean13.py
# Fixed data of the EAN-13 standard (Annex A), built once at import:
# - the binary codes of the digits 0-9 for families A, B and C (table A.2)
# - the first digit deduced from the A/B sequence of digits 2 to 7 (table A.3)

# Convention: N (black) --> 0, B (white) --> 1


import numpy as np


U_MAX = 15 # largest u pre-computed (bigger u, stretch_code(u) is called directly)



# Matrix C contains the binary codes of the digits (0-9) of family C (one row per digit)
C = np.array([[0, 0, 0, 1, 1, 0, 1], 
             [0, 0, 1, 1, 0, 0, 1],
             [0, 0, 1, 0, 0, 1, 1],
             [0, 1, 1, 1, 1, 0, 1],
             [0, 1, 0, 0, 0, 1, 1],
             [0, 1, 1, 0, 0, 0, 1],
             [0, 1, 0, 1, 1, 1, 1],
             [0, 1, 1, 1, 0, 1, 1],
             [0, 1, 1, 0, 1, 1, 1],
             [0, 0, 0, 1, 0, 1, 1]])


h, w = C.shape

# Create matrix A by inverting the bits of each code in C (0 --> 1, 1 --> 0)
A = np.zeros((h, w))
for i in range(h):
    for j in range(w):
        if (C[i][j] == 0):
            A[i][j] = 1
        else:
            A[i][j] = 0
        


# Create matrix B by reading each row of C backwards (left-right mirror of each code)
B = np.zeros((h, w))
for i in range(h):
    for j in range(w):
        B[i][j] = C[i][w - 1 - j]



# Dictionary of the 3 families: key = family letter, value = its 10x7 matrix (row = digit)
dict_families = {"A": A, "B": B, "C": C}


# Dictionary of the first digit: key = A/B sequence of digits 2 to 7 (table A.3)
dict_first_digit = {"AAAAAA": 0, "AABABB": 1, "AABBAB": 2, "AABBBA": 3, "ABAABB": 4,
                    "ABBAAB": 5, "ABBBAA": 6, "ABABAB": 7, "ABABBA": 8, "ABBABA": 9}



# Stretch the codes of the 3 families by u
# Input: u = number of samples per base unit
def stretch_code(u):
    dict_stretched = {}
    for M, code in dict_families.items():                       
        dict_stretched[M] = np.repeat(code, u, axis=1)          
    return dict_stretched



# Pre-computed stretched codes for u = 1 to U_MAX (a dictionary of dictionaries)
dict_stretched_u = {}
for u in range(1, U_MAX + 1):
    dict_stretched_u[u] = stretch_code(u)     