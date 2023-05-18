import numpy as np

def calc_taxicab_var(arr_p, arr_p_var, arr_q, arr_q_var):
    # each summand is divided by 
    numerator = np.abs(arr_p - arr_q)
    denominator = np.sqrt(arr_p_var + arr_q_var)
    summand_signs = get_sign_of_change(arr_p, arr_q)

    # print(numerator)
    # print(denominator)
    # print(np.logical_and(denominator != 0, numerator != 0))
    # print(numerator != 0)
    
    # variance the variance of the denominator should not be zero if the numerator is nonzero
    assert (np.logical_and((denominator != 0), (numerator != 0)) == (numerator != 0)).all()
    denominator[numerator == 0] = 0.01

    tax_varaware_summands = numerator / denominator
    tax_varaware = np.sum(tax_varaware_summands, axis=1)

    return tax_varaware, tax_varaware_summands, summand_signs

"""
Arr[n, m] Arr[n/1, m] -> Arr[n, 1], Arr[n, m]
Taxicab distance between [arr_p] and corresponding row of arr_q (or the single row of arr_q) if it only has one row
Also returns contributions of each column
"""
def calc_taxicab(arr_p, arr_q):
    summand_signs = get_sign_of_change(arr_p, arr_q)
    taxicab_summands = np.abs(arr_p - arr_q)
    taxicab = np.sum(taxicab_summands, axis=-1)
    
    return taxicab, taxicab_summands, summand_signs

"""
`arr_p` is before, `arr_q` is after (makes sense with the letters of the alphabet)
"""
def calc_projections(arr_p, arr_q):
    yty_differences = arr_q - arr_p
    # print(yty_differences)
    diff_p, diff_q = make_shift_row_arrs(yty_differences)
    elementwise_prod = diff_p * diff_q
    # print(elementwise_prod)
    projections = np.sum(elementwise_prod, axis=-1)
    # print(projections)

    return projections


def calc_lp(arr_p, arr_q, p=0.5):
    summand_signs = get_sign_of_change(arr_p, arr_q)
    lp_summands = np.abs(arr_p - arr_q)**p
    lp = np.sum(lp_summands, axis=-1)**(1/p)

    return lp, lp_summands, summand_signs
"""
Arr[n, m] Arr[n/1, m] -> Arr[n, 1], Arr[n, m]
Euclidean distance between [arr_p] and corresponding row of arr_q (or the single row of arr_q) if it only has one row
Also returns contributions of each column
"""
def calc_euclidean(arr_p, arr_q):
    
    euclidean_summands = (arr_p - arr_q)**2
    euclidean = np.sqrt(np.sum(euclidean_summands, axis=-1))
    summand_signs = get_sign_of_change(arr_p, arr_q)
    
    return euclidean, euclidean_summands, summand_signs

"""
Arr[n, m] Arr[n/1, m] -> Arr[n, 1], Arr[n, m]
Hellinger distance between [arr_p] and corresponding row of arr_q (or the single row of arr_q) if it only has one row
Also returns contributions of each column
"""
def calc_hellinger(arr_p, arr_q):
    summand_signs = get_sign_of_change(arr_p, arr_q)
    hellinger_summands = (np.sqrt(arr_p) - np.sqrt(arr_q))**2
    hellinger = np.sqrt(np.sum(hellinger_summands, axis=-1))/np.sqrt(2)
    
    return hellinger, hellinger_summands, summand_signs

"""
Arr[n, m], Arr[n/1, m] -> Arr[n, 1], Arr[n, m]
calculates the kl divergence between rows of arr_p and arr_q. If Arr_q is a single row then it is the KL divergence between 
that row and each row of [arr_p]. That is a single number for each row in [dkl], but the function also returns the contribution
of each column in [dkl_summands]  
"""
def calc_dkl(arr_p, arr_q):
    abs_diff = np.abs(arr_p - arr_q)
    abs_diff = np.where(abs_diff != 0, abs_diff, 1)
    summand_signs = (arr_q - arr_p) / abs_diff
    dkl_summands = arr_p * np.log2(arr_p / arr_q)
    dkl = np.sum(dkl_summands, axis=-1)
    
    return dkl, dkl_summands, summand_signs

"""
Arr -> Arr, Arr
Returns two arrays, one is the array without the first row, the other is the array without the last row.
"""
def make_shift_row_arrs(arr):
    arr_p = arr[:-1,...]
    arr_q = arr[1:,...]
    return arr_p, arr_q

"""
Arr -> Arr
Returns to arrays, one that is equal to the old one, and the other is simply the last row of arr, but maintaining the same dimension
(so one dimension, the row dimension, has only a single element)
"""
def make_past_present_arrs(arr):
    arr_p = arr[:-1,...]
    arr_q = arr[-1:,...]
    
    return arr_p, arr_q

def get_sign_of_change(arr_p, arr_q):
    abs_diff = np.abs(arr_p - arr_q)
    abs_diff = np.where(abs_diff != 0, abs_diff, 1)

    summand_signs = (arr_q - arr_p) / abs_diff

    return summand_signs