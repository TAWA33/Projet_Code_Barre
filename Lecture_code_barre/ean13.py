# ean13.py
# Fixed data of the EAN-13 standard (Annex A), built once when an EAN13Codes object is created
# Convention: N (black) --> 0, B (white) --> 1


import numpy as np



# Codes of family C, copied from table A.2
C_str = ["NNNBBNB",
         "NNBBNNB",
         "NNBNNBB",
         "NBBBBNB",
         "NBNNNBB",
         "NBBNNNB",
         "NBNBBBB",
         "NBBBNBB",
         "NBBNBBB",
         "NNNBNBB"]


# First digit: key = A/B sequence of digits 2 to 7 (table A.3)
dict_first_digit = {"AAAAAA": 0, "AABABB": 1, "AABBAB": 2, "AABBBA": 3, "ABAABB": 4,
                    "ABBAAB": 5, "ABBBAA": 6, "ABABAB": 7, "ABABBA": 8, "ABBABA": 9}



class EAN13Codes:

    # Build all the 0/1 tables once (runs when the object is created)
    def __init__(self, u_max):

        self.u_max = u_max
        self.first_digit = dict_first_digit

        # Convert the N/B strings of C into a 0/1 matrix
        h = len(C_str)
        w = len(C_str[0])
        C = np.zeros((h, w), dtype=int)
        for i in range(h):
            for j in range(w):
                if (C_str[i][j] == "B"):
                    C[i][j] = 1
                else:
                    C[i][j] = 0

        # A = C with inverted bits
        A = np.zeros((h, w), dtype=int)
        for i in range(h):
            for j in range(w):
                if (C[i][j] == 0):
                    A[i][j] = 1
                else:
                    A[i][j] = 0

        # B = each row of C read backwards
        B = np.zeros((h, w), dtype=int)
        for i in range(h):
            for j in range(w):
                B[i][j] = C[i][w - 1 - j]

        # Families: "A", "B", "C" --> 10x7 matrices (row = digit)
        self.families = {"A": A, "B": B, "C": C}

        # Pre-computed stretched codes for u = 1 to u_max
        self.stretched_u = {}
        for u in range(1, u_max + 1):
            self.stretched_u[u] = self.stretch_code(u)


    # Stretch the codes of the 3 families by u (each bar repeated u times)
    def stretch_code(self, u):
        dict_stretched = {}
        for M, code in self.families.items():
            dict_stretched[M] = np.repeat(code, u, axis=1)
        return dict_stretched


    # Give the stretched codes for any u (stored if u <= u_max, computed otherwise)
    def get_codes(self, u):
        if (u <= self.u_max):
            return self.stretched_u[u]
        else:
            return self.stretch_code(u)