
def simulate_constant_model(p0, n):
    return p0

def simulate_drift_model(p0, n, s):
    pt = p0
    for i in range(n):
        pt_flat = pt.flatten()

        sample_counts = np.random.choice(range(len(pt_flat)), size=(s,), replace=True, p=pt_flat)

        rows = sample // pt.shape[1]
        cols = sample % pt.shape[1]

        sample = np.zeros(pt.shape)
        sample = np.add.at(sample, (rows, cols), 1)
        ptplus1 = sample / sample.sum()
        pt = ptplus1
    
    return pt

def simulate_equalization_model(p_array, epsilon):
    p_array_new = (1 - epsilon)*p_array + epsilon / p_array.size
    return p_array_new

def extremifying_evo(p_pair, p_single, epsilon):
    indep_probs = np.matmul(p_single.reshape(1, len(p_single)), p_single.reshape(len(p_single), 1))
    inc = (p_pair / indep_probs - 1)*epsilon
    p_pair_new = inc_and_norm(p_pair_new, inc)

    return p_pair_new

def time_correlated_noise(p_array_0, p_array_1, sd):
    noise = np.random.normal(loc=(p_array_1 - p_array_0), scale=sd)
    new_array = inc_and_norm(p_array_1, noise)
    return new_array

def variance_prop_noise(p_array, N, sd_coeff):
    noise = np.random.normal(loc=0, scale=sd_coeff*np.sqrt(p_array(1 - p_array)/N))
    new_array = inc_and_norm(p_array, noise)
    return new_array


def normal_noise(p_array, sd):
    noise = np.random.normal(loc=0, scale=sd, size=p_array.shape)
    new_array = inc_and_norm(p_array, noise)
    return new_array 

def inc_and_norm(arr, noise):
    arr_inc = arr + noise
    arr_inc_and_norm = arr_inc / arr_inc.sum()

    return arr_inc_and_norm


