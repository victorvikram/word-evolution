import numpy as np 
import context_distribs as cd
import matplotlib.pyplot as plt
import sys
import load_data as ld

def print_all_local_vars(local_vars, thresh=0):
    longest_var = max([len(elt[0]) for elt in local_vars])

    for var, obj in local_vars:
        size = ld.get_full_size(obj) / 1024 / 1024
        if size > thresh:
            print(f"{var:<{longest_var}} : {size:.2f}")

"""
Dict(String <> Int) Arr[n,] ->
vector[i,] is a certain quantity corresponding to word i. The function prints the words along with their corresponding quantities
in ascending order
"""

def print_word_quantity_vector(word_dict, vector):
    sorted_indices = np.argsort(vector)
    
    for i in range(len(vector)):
        ind = sorted_indices[i]
        word = word_dict[ind]
        val = vector[ind]
        print(f'{ind}, {word}: {val}')

def print_word_window_quantity_vector_flat(word_dict, window_dict, vector):
    num_words = len(word_dict) // 2
    num_windows = len(window_dict) // 2
    assert (num_words * num_windows == vector.shape[0])
    sorted_indices = np.argsort(-vector)

    for i in range(len(vector)):
        ind = sorted_indices[i]
        word_ind = ind % num_words
        window_ind = ind // num_words
        print(f'{ind}, {window_dict[window_ind]}, {word_dict[word_ind]}, val: {vector[ind]}')

def printCooWordArr(word_dict, cooArr):
    def printElt(a):
        print(f"{[word_dict[elt] for elt in a[:-1]]}: {a[-1]}")
    
    fullArr = np.concatenate((cooArr.coords, [cooArr.data]), axis=0)
    np.apply_along_axis(printElt, axis=0, arr=fullArr)

def printWordTupleDict(dct, word_dict):
    for wind, subdct in dct.items():
        print(wind, "--")
        for key, val in subdct.items():
            print(f"{[word_dict[i] for i in key]}: {val}")

"""
Dict(String <> Int) Arr[m, n] -> 
ind_arr is an array whose rows correspond time windows, or specific situations. Each row contains a series of indices corresponding to relevant
words for that situation. This function prints the words corresponding to those indices
"""
def print_word_indices(word_dict, ind_arr, window_dict=None):

    for i, row in enumerate(ind_arr):
        print(window_dict[i] if window_dict is not None else i)
        print([word_dict[ind] for ind in row])

        
def printDict(dct, fxn=(lambda x: x), sortByKey=False, sortByVal=False):

    if sortByKey:
        iterator = sorted(dct.items(), key=(lambda x: x[0])) # key is first element of the tuple
    elif sortByVal:
        iterator = sorted(dct.items(), key=(lambda x: x[1])) # key is the second element of the tuple
    else:
        iterator = dct.items()

    for key, val in iterator:
        print(fxn(key), val)

"""
Dict(Dict(List(Int))) List(Any) -> Figure Axes
Each entry Dict[key] corresponds to lines on subplots labeled with "key". For instance, each might correspond to a word that is plotted
on a number of different subplots.
Each entry Dict[key][subkey] is a list corresponding to the line for "key" on each different subplot, corresponding to "subkey". For instance,
the subplots might correspond to different time series for a particular word: frequency, change in meaning, etc.
subkeys gives which keys in the subdicts should be plotted as a subplot. This is why there are len(subkeys) subplots.
And each subplot has a line corresponding to a key in the dict. 
"""
def plotFromDict(dct, subkeys=None):
    import matplotlib.pyplot as plt 
    
    if subkeys is None: 
        val = next(iter(dct.values()))
        subkeys = list(val.keys())
    
    fig, axs = plt.subplots(len(subkeys))

    for key, val in dct.items(): 
        for i, subkey in enumerate(subkeys):
            axs[i].plot(range(len(val[subkey])), val[subkey], label=key)
    
    return fig, axs


def get_average_bins(xs, ys, bins):
    min_val = min(xs)
    bin_size = (max(xs) - min_val) / bins

    infos = []
    for i in range(bins):
        start = min_val + i * bin_size
        end = start + bin_size

        relevant_indices = np.nonzero(np.logical_and(xs >= start, xs < end))
        # print(relevant_indices)
        relevant_ys = ys[relevant_indices]
        val = relevant_ys.sum() / len(relevant_ys)

        infos.append((start, end, val))
    
    return infos

def plot_averages_on_scatter(ax, xs, ys, bins):
    infos = get_average_bins(xs, ys, bins)
    for info in infos:
        ax.plot([info[0], info[1]], [info[2], info[2]], color="black")
    
    return ax

def scatter(x, y, title="", include_regression=False, average_bins=None, skip=1, kwargs={"s": 0.5, "alpha": 0.3}):
    x = np.array(x)
    y = np.array(y)
    fig, ax = plt.subplots(1)
    x = x[::skip]
    y = y[::skip]

    ax.scatter(x, y, **kwargs)
    ax.set_title(title)

    if include_regression:
        olsres, y_pred = cd.do_regression(x, y)
        ax.plot(x, y_pred, color="red")
    
    if average_bins is not None:
        plot_averages_on_scatter(ax, x, y, bins=average_bins)
    
    return fig, ax

def plot_column_change(matrix, title=""):
    changes_from_beg = matrix - matrix[0:1,:]
    fig, ax = plt.subplots(1)
    ax.plot(range(changes_from_beg.shape[0]), changes_from_beg)
    ax.grid()
    ax.set_title(title)
    return fig, ax

