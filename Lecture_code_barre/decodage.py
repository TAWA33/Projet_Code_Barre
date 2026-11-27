import numpy as np
from Lecture_code_barre.ean13 import EAN13Codes


# Extract the partial signature containing the 12 digits that need to be decoded
def extract_partial_signature(signature, u):
    s_p = np.zeros((12, 7*u), dtype=int)
    first_part = signature[3*u : 45*u]
    second_part = signature[50*u : 92*u]
    concatenated = np.concatenate((first_part, second_part))
    for i in range(0, 12):
            s_p[i] = concatenated[7*i*u : 7*(i+1)*u]
    return s_p



# Decode one digit: compare its partial signature (7u samples) with the codes of the allowed families
# (["A", "B"] for the left digits, ["C"] for the right ones) and keep the closest one.
def decode_digit(digit_signature, codes_u, families):
    list_codes = []
    for f in families:
        list_codes.append(codes_u[f])
    candidates = np.concatenate(list_codes, axis=0)
    c = np.sum(np.abs(digit_signature - candidates), axis=1)
    min_index = np.argmin(c)
    digit = min_index % 10
    family = families[min_index // 10]
    score = c[min_index]
    return (digit, family, score)



# Decode the 12 digits of the barcode from the partial signatures s_p (12 x 7u, one row per digit)
# Left digits are compared with families A and B, right digits with family C
def decode_digits(s_p, codes, u):
    codes_u = codes.get_codes(u)
    digits = np.zeros(12, dtype=int)
    families = [""] * 12
    scores = np.zeros(12)
    for i in range(12):
        if i <= 5:
            (digits[i], families[i], scores[i]) = decode_digit(s_p[i], codes_u, ["A", "B"])
        else:
            (digits[i], families[i], scores[i]) = decode_digit(s_p[i], codes_u, ["C"])
    return (digits, families, scores)



# Decode the first digit from the A/B sequence of digits 2 to 7
def decode_first_digit(families, dict_first_digits):
    concatenated = "".join(families[0:6])
    if concatenated in dict_first_digits:
        first_digit = dict_first_digits[concatenated]
        return first_digit
    else:
        return None



# Check the control key for the 13th digit.
def check_control_key(digits_list):
    total = 0
    for i in range(len(digits_list) - 1):
        if i % 2 == 0: # the first digit has the index 0 in the list, but has the rank 1 which is odd
            total += digits_list[i]
        else:
            total += 3 * digits_list[i]
    unit_digit = total % 10
    if (10 - unit_digit) % 10 == digits_list[-1]:
        return True
    else:
        return False



# Count the edges (black <-> white switches) in a binarized signature
# A correct EAN-13 reading always has exactly 58 edges (30 black bars, 29 white spaces between them)
def count_edges(binarized_signature):
    differences = np.diff(binarized_signature, n=1)
    count = 0
    for i in range(len(differences)):
        if differences[i] == 1 or differences[i] == -1:
            count += 1
    return count