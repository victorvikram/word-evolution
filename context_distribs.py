import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import re
import distances as dist
import ndd
import sparse
import math
import multiprocessing
import sys
import random
import psutil
import os
import pickle as pkl

from collections import defaultdict
import word_counting as wc
import display as d
import load_data as ld

"""
Dict(String: Int) List(List(String)) Series Dict(String <> Int) Int Int -> Arr(len(words)*np.unique(window_col), top_m)
Creates an array where each row corresponds to a window-word, and each col corresponds to a word. Arr[i, j] is the probability
that a given word in the context of window-word i is word j. So each row is a distribution around a particular focal word
in a given window of time. 
"""
def genWordWindowContextDistribs(cat, focal_word_dict, wordcount_window, frame=10, top_m=30000, common_sample=None, symmetric=False, parallel=True):
    winds = ld.get_keys_for_name(os.path.join("objects", f"{cat}"), f"{cat}_speech_dict", int)

              
    inputs = [(cat, focal_word_dict, wordcount_window[wind], wind, frame, top_m, common_sample, symmetric) for wind in winds]
    inputs = sorted(inputs, key=lambda x: x[3])

    if parallel:
        with multiprocessing.Pool() as pool:
            word_context_counts = pool.starmap(gen_window_context_distrib, inputs)
    else:
        word_context_counts = list(map(lambda args: gen_window_context_distrib(*args), inputs))  
    
    word_context_counts = np.array(word_context_counts)
    # print(word_context_counts.shape)
    
    totals = word_context_counts.sum(axis=-1, keepdims=True)
    print(totals)
    # variance of multinomial is n*pi*(1 - pi)
    word_context_pcts = word_context_counts / totals
    word_context_variances = totals * word_context_pcts * (1 - word_context_pcts)
    word_context_pctvar = word_context_variances / (totals*totals) # to get variance of individual entries I treat n as a constant. Not sure if this is a sin.
    
    return word_context_pcts, word_context_pctvar, word_context_counts, word_context_variances, focal_word_dict

def gen_window_context_distrib(cat, focal_word_dict, wordcount_window, wind, frame=10, top_m=30000, common_sample=None, symmetric=False):
    # gets the index of the context word
    num_words = len(focal_word_dict)
    num_context_words = top_m if symmetric is False else num_words

    word_context_counts = np.zeros((num_words, num_context_words))

    speech_list = ld.read_by_key(f"{cat}_speech_dict", wind, os.path.join("objects", cat))

    def context_word_ind(word):
        if symmetric:
            return focal_word_dict[word] if word in focal_word_dict else None
        else:
            if word < top_m:
                return word
            else:
                return None
    
    num_speeches = len(speech_list)
    for j, speech in enumerate(speech_list):
        if j % 100000 == 0:
            print(f"{wind}: {j}/{num_speeches}")
        
        current_context = []

        for i in range(frame + 1):
            if i < len(speech):
                word_index = context_word_ind(speech[i])
                current_context.append(word_index)
            else:
                current_context.append(None)

        context_ind = 0
        
        for i, word in enumerate(speech):
            if word in focal_word_dict:
                word_index = focal_word_dict[word]
                if common_sample is not None:
                    # print("wind", "word", wind, word)
                    # print("common sample", common_sample)
                    # print("apperances",wordcount_window[word])
                    # number of sampled words should be about equal to min_count
                    sample_prob = common_sample / wordcount_window[word]
                    # print("probability", sample_prob)
                else:
                    sample_prob = 1
                
                if np.random.rand() < sample_prob:
                    # word_window_index = word_index * n_windows + window_index
                    collapsed_context = [item for i, item in enumerate(current_context) if item is not None and i != context_ind]
                    
                    # print(focal_word, word_window_index, collapsed_context)
                    np.add.at(word_context_counts, (word_index, collapsed_context), 1) # replaced next line with this one because it coutns repeats
                    # word_context_counts[window_index, word_index, collapsed_context] += 1
                
            # if we are not smushed against the beginning of the list
            if context_ind == frame:
                current_context = current_context[1:]
            else:
                context_ind += 1
            
            # if we are not smushed against the end of the list
            if i + 1 + frame < len(speech):
                new_ind = context_word_ind(speech[i + 1 + frame])
                current_context.append(new_ind)

    return word_context_counts

"""
Dict(String <> Int) List(List(String)) Series Dict(String <> Int) Dict(Int <> Any) Arr[num_windows, num_words] Int Int Bool Int Bool
->
Arr[num_windows * num_focal_words, num_words] * 4 Dict(String <> Int) * 2

Takes a dictionary of [words], then finds the distribution of each word's context for each window given in window_col. [sorted_word_dict]
is the dictionary connecting words to keys in [wordcount_window], [frame] is the size of the context, [top_m] is the number of context words to sample,
sample_equally requires all [words] have an equal number of contexts sampled (so distribution variabilities are commensurate), min_count only includes
words in [words] whose contexts all have a count above [min_count], and [symmetric] means the only context words counted are the words in [words]

Returns an unrolled array with each row corresponding to the distribution of the context of a particular focal word-window, and each column is a context
word. The first returned dict is the focal word dictionary, the second one is the dictionary of context words (which may be equal to the focal word dictionary
if [symmetric] is True
"""
def genWordWindowContextDistribsSpecific(words, list_of_speech_lists, window_col, sorted_word_dict, window_dict, wordcount_window, frame=10, top_m=30000, sample_equally=False, min_count=None, symmetric=False):
    if sample_equally:
        word_list = [word for word in words if isinstance(word, str)]
        common_sample, _, indices_to_remove = getLeastCommonWordWindow(wordcount_window, sorted_word_dict, words_of_interest=word_list, min_threshold=min_count)
        # print(least_common_word)
        # print(indices_to_remove)
    else:
        indices_to_remove = []
        common_sample = None
    
    focal_word_dict = {word: ind for word, ind in words.items() if ind not in indices_to_remove}
    focal_word_dict.update({ind: focal_word for focal_word, ind in words.items()})
    num_words = len(focal_word_dict) // 2
    return genWordWindowContextDistribs(focal_word_dict, num_words, list_of_speech_lists, window_col, sorted_word_dict, window_dict, wordcount_window, frame, top_m, common_sample, symmetric=symmetric)


"""
List(List(String)) Series Dict(String <> Int) Dict(Int <> Any) Arr[num_windows, num_words] Int Int Int Int Bool Int Bool
->
Arr[num_windows * num_focal_words, num_words] * 4 Dict(String <> Int) * 2

Takes a [start_ind] and an [end_ind], then finds distribution of the context of each word corresponding to thise indices in [sorted_word_dict], for
each window. [sorted_word_dict] is the dictionary connecting words to keys in [wordcount_window], [frame] is the size of the context, [top_m] is the number of context words to sample,
sample_equally requires all [words] have an equal number of contexts sampled (so distribution variabilities are commensurate), min_count only includes
words in [words] whose contexts all have a count above [min_count], and [symmetric] means the only context words counted are the words in [words]

Returns an unrolled array with each row corresponding to the distribution of the context of a particular focal word-window, and each column is a context
word. The first returned dict is the focal word dictionary, the second one is the dictionary of context words (which may be equal to the focal word dictionary
if [symmetric] is True
"""
def genWordWindowContextDistribsStartToEnd(cat, frame=10, start_ind=0, end_ind=3000, top_m=30000, sample_equally=True, min_count=None, symmetric=False):
    wordcount_window, sorted_word_dict, _ = ld.load_counts(cat)
    if sample_equally:
        common_sample, _, indices_to_remove = getLeastCommonWordWindow(wordcount_window, sorted_word_dict, words_of_interest=None, start_ind=start_ind, end_ind=end_ind, min_threshold=min_count)
    else:
        indices_to_remove = []
        common_sample = None
    
    added_counter = -1
    focal_word_dict = {i: (added_counter := added_counter + 1) for i in range(start_ind, end_ind) if i - start_ind not in indices_to_remove}
    print("focal word dict", focal_word_dict, indices_to_remove)

    return genWordWindowContextDistribs(cat, focal_word_dict, wordcount_window, frame, top_m, common_sample, symmetric)

"""
co-occurs within [frame] words (adjacent words are 1 word away, and so on...) For a frame of 3, then, we would include focal_word x x context_word
"""
def genNWiseCooccurenceDistribs(speech_dir, winds, n, fract=None, frame=20, num_focal_words=500, num_context_words=1000, normalize=False, parallel=True, par_memo=False, tuple_dict=None, tuple_thresh=1, return_dict=False):

    if tuple_dict is not None and tuple_thresh >= 2:
        print(f"before paring down", sys.getsizeof(tuple_dict))
        tuple_dict = {tup: num for tup, num in tuple_dict.items() if num >= tuple_thresh}
        print(f"max size of tuple dict for {n}-tuples is", sys.getsizeof(tuple_dict))

    if par_memo:
        memoTuple = multiprocessing.Manager().dict()
        memoFrame = multiprocessing.Manager().dict()
    else:
        memoTuple = {}
        memoFrame = {}
    
    coOccurDict = {}
    speechesExt = [(speech_dir, n, fract, frame, num_focal_words, num_context_words, wind, tuple_dict, memoTuple, memoFrame, return_dict) for wind in winds]
    print("length of the extended speech list", len(speechesExt))

    if parallel:
        with multiprocessing.Pool() as pool:
            window_arr_dicts = pool.starmap(genNWiseCooccurenceDistribsWind, speechesExt)
    else:
        window_arr_dicts = list(map(lambda args: genNWiseCooccurenceDistribsWind(*args), speechesExt))  
    
    if return_dict:
        window_cooccur_arr = [elt[0] for elt in window_arr_dicts]
        window_cooccur_dict = [elt[1] for elt in window_arr_dicts]

    
        total_cooccur_dct = {}
        for wind_dict in window_cooccur_dict:
            incrementWithDict(total_cooccur_dct, wind_dict)

        wind_cooccur_dct = dict(zip(winds, window_cooccur_dict))
        wind_cooccur_arr = dict(zip(winds, window_cooccur_arr))
    
        if normalize:
            for wind, arr in coOccurDict.items():
                tot = np.sum(arr.data)
                coOccurDict[wind] = arr / tot
    
        return wind_cooccur_arr, wind_cooccur_dct, total_cooccur_dct
    
    else:
        return None

"""
Takes a random sample of [speech_list_dict] and generates the set of triples and pairs that occur in that random sample. Returns those dictionaries. The returned 
[pair_dict] and [trip_dict] are compatible (that is, every triple has its constituent *focal* pairs present in the pair_dict -- a focal pair is a pair containing
one of the focal words). Thus the distribution comparison should run smoothly.
"""
def gen_pair_trip_dict(speech_dir, winds, fract, frame=20, num_focal_words=500, num_context_words=1000, normalize=False, parallel=False, par_memo=True):
    _, _, trip_dict = genNWiseCooccurenceDistribs(speech_dir, winds, 3, fract, frame, num_focal_words, num_context_words, normalize, parallel, par_memo, tuple_dict=None, return_dict=True)
    _, _, pair_dict = genNWiseCooccurenceDistribs(speech_dir, winds, 2, fract, frame, num_focal_words, num_context_words, normalize, parallel, par_memo, tuple_dict=None, return_dict=True)
    match_pair_trip_dict(pair_dict, trip_dict, num_focal_words, num_context_words)

    return pair_dict, trip_dict

"""
Takes a random sample of [speech_list_dict] and generates a dictioanry of [n]-tuples [tuple_dict], that occurs in that random sample
"""
def gen_tuple_dict(speech_list_dict, n, sorted_word_dict, frame=20, num_focal_words=500, num_context_words=1000, normalize=False, parallel=False, par_memo=True):
    _, _, tuple_dict = genNWiseCooccurenceDistribs(sample_speech_list_dict, n, sorted_word_dict, frame, num_focal_words, num_context_words, normalize, parallel, par_memo, tuple_dict=None)

    return tuple_dict

"""
this function takes a [pair_dict] and a [trip_dict], which are dictionaries of 2-tuples and 3-tuples, along with their counts. It adds all necessary pairs to the [pair_dict]
to make them compatible. They are compatible if all constituent pairs of a triplet *that contain the focal word* are present in the pair dict. Mutates!
"""
def match_pair_trip_dict(pair_dict, trip_dict, num_focal_words, num_context_words):
    for trip in trip_dict.keys():
        candidate_tuples = [(trip[0], trip[1]), (trip[0], trip[2]), (trip[1], trip[2])]
        candidate_tuples = [tup for tup in candidate_tuples if min(tup) < num_focal_words and max(tup) < num_context_words]

        for tup in candidate_tuples:
            if tup not in pair_dict:
                pair_dict[tup] = 1
    
"""
takes [speech_lists], which is a list of word lists (each constituent word lists is a speech), and counts up all the [n]-tuples. Can pass dictionaries [memoTuple] and [memoFrame]
to memoize either the tuples in a list of a particular length or the tuples within [frame] distance of each other in a list of a particular length. Returns a COO representation
of the k tuples (which is coordinates and their values, since the array is likely to be extremely sparse), and also returns a dictionary with the k-tuples and their counts. 
"""
def genNWiseCooccurenceDistribsWind(speech_dir, n, fract=None, frame=20, num_focal_words=500, num_context_words=1000, wind=0, tuple_dict=None, memoTuple={}, memoFrame={}, return_dict=False):
    pid = os.getpid()
    process = psutil.Process(pid)

    speech_lists = ld.read_by_key(f"{speech_dir}_speech_dict", wind, os.path.join("objects", speech_dir))

    if fract is not None:
        sample_length = round(len(speech_lists) * fract)
        speech_lists = random.sample(speech_lists, sample_length)

    
    # totalDict = {}
    tuple_counts_dict = tuple_dict if tuple_dict is not None else {}
    for key in tuple_counts_dict:
        tuple_counts_dict[key] = 0

    def tuple_criterion(tup):
        relevant_tuple = tup in tuple_dict if tuple_dict is not None else True
        contains_focal = min(tup) < num_focal_words
        restricted_context = max(tup) < num_context_words

        return relevant_tuple and contains_focal and restricted_context

    for j, speech in enumerate(speech_lists):
        if j % 10000 == 0:
            print(f"{wind}: {j}/{len(speech_lists)}")
            """
            local_vars = list(locals().items())
            d.print_all_local_vars(local_vars)
            """
            memory_info = process.memory_info().rss
            print(f"total memory {memory_info / 1024 / 1024:.2f}")

        candidate_tuples = findKTuplesFrameLst(n, speech, frame, memoTuple, memoFrame, dct=True)
        filtered_tuples = {tup: count for tup, count in candidate_tuples.items() if tuple_criterion(tup)}

        incrementWithDict(tuple_counts_dict, filtered_tuples)
        memoFrame = {} # don't save the memoframe because it gets huge
    
    # print({key: val for i, (key, val) in enumerate(memoTuple.items()) if i < 10})
    # print({key: val for i, (key, val) in enumerate(memoFrame.items()) if i < 10})

    tuple_counts_arr = makeSparseArr(tuple_counts_dict, num_context_words, n)

    if not return_dict:
        with open(os.path.join("objects", speech_dir, f"{speech_dir}_{n}tuple_dict_{wind}.pkl"), "wb") as f:
            pkl.dump(tuple_counts_dict, f)
        
        with open(os.path.join("objects", speech_dir, f"{speech_dir}_{n}tuple_arr_{wind}.pkl"), "wb") as f:
            pkl.dump(tuple_counts_arr, f)

        return None
    
    else:
        return tuple_counts_arr, tuple_counts_dict
"""
generates a COO sparse array from a list of coordinates
"""
def convertLstToSparseArr(lst, k):
    if len(lst) == 0:
        coo = sparse.COO(coords=np.zeros((k, 0)), data=[], shape=(10,)*3)
    else:
        repeatCoords = np.transpose(lst)
        coords, data = np.unique(repeatCoords, axis=1, return_counts=True)
        coo = sparse.COO(coords=coords, data=data)
    
    return coo

"""
turns a list into a dct with all the unique elements of the list as keys, and the counts of each unique element as the values
"""
def convertLstToDct(lst):
    dct = {}
    incrementAll(dct, lst)
    return dct

"""
finds all k-tuples in a lsit that are within a window of size [frame]. That is, w1 _ w2 are within a frame of 
size 3, but not 2 (even though they are two apart from each other).
"""
def findKTuplesFrameLst(k, lst, frame, memoTuple, memoFrame, dct=True, sparseArr=False):
    kTupleIndices = findKTuplesFrame(k, len(lst), frame, memoTuple, memoFrame)
    kTuples = [tuple(sorted([lst[i] for i in tup])) for tup in kTupleIndices]

    if dct:
        kTuples = convertLstToDct(kTuples)
    elif sparseArr:
        kTuples = convertLstToSparseArr(kTuples, k)

    return kTuples

"""
finds the indices of all [k]-tuples for a list of size [n]. can also pass it a dictionary [memoTuple] whose keys are (k, n) pairs, since very often
a particular request will have already been done/
""" 
def findKTuples(k, n, memoTuple):

    if (k, n) in memoTuple:
        # print("retrieved memoized")
        return memoTuple[(k, n)]

    if k == 1:
        kTuplesAll = [(i,) for i in range(n)]
        km1TuplesNotLast = [()]
    elif n > k:
        km1TuplesNotLast = findKTuples(k - 1, n - 1, memoTuple)
        kTuplesNotLast = findKTuples(k, n - 1, memoTuple)

        kTuplesLast = [tup + (n - 1,) for tup in km1TuplesNotLast]
        kTuplesAll = kTuplesNotLast + kTuplesLast

        # assert math.comb(n, k) == len(kTuplesAll)
        # assert len(set(kTuplesAll)) == len(kTuplesAll)
        # print("tests passed!")
    elif k > n:
        kTuplesAll = []
    elif k == n:
        kTuplesAll = [tuple(range(n))]
    

    memoTuple[(k, n)] = kTuplesAll
    return kTuplesAll

"""
finds the indices of [k]-tuples that are within a window of size [frame], for a list of size [n]. can pass it memoized values since
there are many calls to this function for the same set of parameters.
"""
def findKTuplesFrame(k, n, frame, memoTuple, memoFrame):
    if (k, n, frame) in memoFrame:
        return memoFrame[(k, n, frame)]
    
    allKTuples = []
    startIndex = 0
    endIndex = min(frame, n) - 1
    
    kTuples = [tuple(sorted(tup)) for tup in findKTuples(k, endIndex, memoTuple)]
    km1Tuples = findKTuples(k - 1, endIndex, memoTuple)
    allKTuples += kTuples
    # print(endIndex, kTuples, km1Tuples)
    
    while endIndex < n:
        kTuples = [tuple(sorted(tup + (endIndex,))) for tup in km1Tuples] # add 
        # print(endIndex, kTuples, km1Tuples)
        allKTuples += kTuples
        km1Tuples = [tuple([endIndex if elt == startIndex else elt for elt in tup]) for tup in km1Tuples]
        startIndex += 1
        endIndex += 1
    
    memoFrame[(k, n, frame)] = allKTuples

    return allKTuples

def calcDistribDiffsForEachWordWindow(coOccurTripDict, coOccurPairDict):
    allDistances = []

    for wind in coOccurTripDict.items():
        distances, indepDistribs, trueDistribs = calcDistribDiffsForEachWord(coOccurTripDict[wind], coOccurPairDict[wind])
        allDistances.append(distances)
    
    return allDistances
        
def distrib_diffs_all_windows(cooccur_trip, cooccur_pair, num_focal_words=500):
    pair_trip_tups = [(trip_arr, pair_arr, num_focal_words) for trip_arr, pair_arr in zip(cooccur_trip.values(), cooccur_pair.values())]
    
    with multiprocessing.Pool() as pool:
        results = pool.map(calcDistribDiffsForEachWord, pair_trip_tups)
    
    return results
    
def calcDistribDiffsForEachWord(coOccurTrip, coOccurPair, num_focal_words=500):
    np.unique(coOccurTrip.coords)
    indepDistribs = {}
    trueDistribs = {}
    distances = np.zeros((max(coOccurTrip.shape),))

    relevant_words = np.unique(coOccurTrip.coords)
    relevant_words = relevant_words[relevant_words < num_focal_words]

    for wordInd in relevant_words:
        print(wordInd, end="\r")
        tripDistrib, indepTripDistrib = getDistribForWordI(wordInd, coOccurTrip, coOccurPair)
        indepDistribs[wordInd] = indepTripDistrib
        trueDistribs[wordInd] = tripDistrib

        # [QUESTION] what is the right distance metric here? because it is true that words that show up less will have higher values in the prob
        # distribution which consequently have more variance (portending that they will bias higher) but they also have fewer slots filled, so 
        # fewer terms in the sum (portending that they will bias lower). So it is unclear which bias will win out... I can probably calculate this tho
        # or have a null model?
        distances[wordInd], _, _ = dist.calc_taxicab(indepTripDistrib, tripDistrib)
    
    return distances, indepDistribs, trueDistribs

def getDistribForWordI(i, coOccurTrip, coOccurPair):
    # print("triple")
    filteredTripCoords, filteredTripData = selectCoordsWithI(i, coOccurTrip)
    # print("pair")
    filteredPairCoords, filteredPairData = selectCoordsWithI(i, coOccurPair)

    tripDistrib = filteredTripData / filteredTripData.sum()
    pairDistrib = filteredPairData / filteredPairData.sum()
    # print("trip distrib \n", tripDistrib)
    # print("pair distrib \n", pairDistrib)
    # print("triple")
    collapsedTripCoords = deleteOneAndCollapse(filteredTripCoords, i)
    # print("pair")
    collapsedPairCoords = deleteOneAndCollapse(filteredPairCoords, i)
    
    # print("shapes of trip and pair coords", collapsedTripCoords.shape, collapsedPairCoords.shape)
    # print(list(np.isin(collapsedTripCoords, collapsedPairCoords)))
    pairProbArr = selectCorrespondingElt(collapsedTripCoords, collapsedPairCoords.flatten(), pairDistrib.flatten())
    
    # print("pair prob arr", pairProbArr)
    indepTripDistrib = np.prod(pairProbArr, axis=0)
    # print(indepTripDistrib)

    return tripDistrib, indepTripDistrib

def selectCorrespondingElt(keyArray, keys, vals):
    keySorted = np.argsort(keys)
    keyIndices = np.searchsorted(keys[keySorted], keyArray)
    valArray = vals[keySorted][keyIndices]

    return valArray

def deleteOneAndCollapse(arr, val):
    valIndices = np.where(arr == val)
    colSort = np.argsort(valIndices[1])
    reOrderedCols = valIndices[1][colSort]
    reOrderedRows = valIndices[0][colSort]
    colIndices = np.unique(reOrderedCols)
    # print("valIndices \n", valIndices)
    firstIndices = np.searchsorted(reOrderedCols, colIndices)
    rowIndices = reOrderedRows[firstIndices]
    # print("colIndices \n", colIndices)
    # print("rowIndices \n", rowIndices)

    arrCollapsedFlat = np.delete(arr.ravel("F"), colIndices*arr.shape[0] + rowIndices)
    arrCollapsed = np.reshape(arrCollapsedFlat, (arr.shape[0] - 1, -1), "F")

    # print("collapsed \n", arrCollapsed)

    return arrCollapsed


def selectCoordsWithI(i, cooArray):
    mask = np.any((cooArray.coords == i), axis=0)
    indices = np.where(mask)[0]
    # print("initial coords \n", cooArray.coords)
    # print("initial data \n", cooArray.data)
    filteredCoords = cooArray.coords[:, indices]
    filteredData = cooArray.data[mask]
    # print("filtered coords \n", filteredCoords)
    # print("filtered data \n", filteredData)
    return filteredCoords, filteredData

def makeSparseArr(windowDict, num_context_words, n):
    tup_list = [tup for tup in windowDict.keys()]
    val_list = [val for val in windowDict.values()]

    coords = np.array(tup_list, dtype=int).transpose()
    vals = np.array(val_list, dtype=int)

    s = sparse.COO(coords, data=vals, shape=(num_context_words,)*n)

    return s

def indep_estimate_from_two_tuple_arr(two_tuple_arr, wordcount_for_window):
    wordcount_pct = wordcount_for_window / wordcount_for_window.sum()
    pair_pcts = wordcount_pct[two_tuple_arr.coords]
    indep_probs = np.prod(pair_pcts, axis=0)
    return indep_probs


def incrementAll(dct, keys):
    for key in keys:
        incOrAssignKey(dct, key, 1)

def incrementWithDict(baseDct, incDct):
    for key, val in incDct.items():
        incOrAssignKey(baseDct, key, val)

def incOrAssignKey(dct, key, val):
    if key in dct:
        dct[key] += val
    else:
        dct[key] = val

    
def getGroups(lst, n):
    if n == 1:
        return getSingles(lst)
    elif n == 2:
        return getPairs(lst)

def getSingles(lst):
    return [[i] for i in lst]

def getPairs(lst):
    return [[item, other] for i, item in enumerate(lst) for other in lst[i + 1:]]


"""
String List(List(String)) Series Dict(String <> Int) Int -> Array[np.unique(window), # of words] Dict(Int: Any)
context_array[i, j] is the probability of a given word in the context of i being word j. Also returns a dictionary with
the indices of each window corresponded with the name of the window.
"""
def genWindowContextDistribs(word, list_of_speech_lists, window_col, word_dict, window_dict, frame=10):
    context_array = np.zeros((len(window_col.unique()), len(word_dict.keys()) // 2))
    
    for i, speech in tqdm(enumerate(list_of_speech_lists)):
        curr_window = window_col.iloc[i]
        
        for ind_match in getAllWordMatchIndices(word, speech):
            context = getRangeFromIterable(speech, ind_match, frame)
            # print(context)
            # y = input("hi")
            
            for context_word in context:
                context_array[window_dict[curr_window], word_dict[context_word]] += 1
    
    context_array = context_array / context_array.sum(axis=1, keepdims=True)

    return context_array

"""
Iterable Int Int -> Iterable
Returns a window of [range_size] before and after a given element in an iterable
"""
def getRangeFromIterable(iterable, index, range_size):
    rng_ind = (max(index - range_size, 0), min(index + range_size, len(iterable)))
    rng = iterable[rng_ind[0]:rng_ind[1]]
    
    return rng

"""
String List(String) -> List(Int)
returns all indices of word in lst
"""
def getAllWordMatchIndices(word, lst):
    return [word_ind for word_ind, word_val in enumerate(lst) if word_val == word]

"""
DataFrame String ---> String, String, Int, String
yields all the next occurences of a word in the speech column, along with the 50 characters
before and after.
"""
def getNextContextOfWord(speeches, word, number_required=1):
    regex = r'\b' + re.escape(word) + r'\b'
    
    for i, row in speeches.iterrows():  
        speech = row["speech"]
        
        for match in re.finditer(regex, speech):
            ind = match.start(0)
            context = getRangeFromIterable(speech, ind, 50)

            if len(re.findall(regex, context)) >= number_required:
                yield row["speaker"], row["chamber"], row["date"], context
    

"""
Dict(String: Int) Array[len(words_of_interest)*len(window_dict)/2] Dict(String <> Int) Dict(Any <> Int) -> Dict(String: Dict)
Takes a series of words_of_interest which are each marked by an index, and calculates the changes to the distribution from year
to year and from year to present. Returns a dictionary with each word as a key.
"""
def gatherDataFromContextDistribs(words_of_interest, word_context_distrib, word_dict, window_dict, word_count_vec=None, word_context_var=None, distance_metric=dist.calc_taxicab):

    num_windows = len(window_dict) // 2
    differences = {}
    print(word_context_distrib.shape)

    for word in words_of_interest:
        
        word_index = words_of_interest[word]
        start_index = word_index * num_windows
        end_index = start_index + num_windows
        print(start_index, end_index)

        # doesn't work yet
        if word_context_var is not None:
            yty, yty_summands, yty_signs = time_series_differences_var_aware(word_context_distrib[start_index:end_index,:], 
                                                                    word_context_var[start_index:end_index,:], comp_func=dist.make_shift_row_arrs)
            ytp, ytp_summands, ytp_signs = time_series_differences_var_aware(word_context_distrib[start_index:end_index,:], 
                                                                    word_context_var[start_index:end_index,:], comp_func=dist.make_past_present_arrs)
        else:
            yty, yty_summands, yty_signs = time_series_differencing(word_context_distrib[start_index:end_index,:], comp_func=dist.make_shift_row_arrs, distance_metric=distance_metric)
            ytp, ytp_summands, ytp_signs = time_series_differencing(word_context_distrib[start_index:end_index,:], comp_func=dist.make_past_present_arrs, distance_metric=distance_metric)

        # seems to overcompensate
        if word_count_vec is not None:
            print(yty)
            print(word_count_vec[start_index:end_index - 1])
            yty = yty * np.sqrt(word_count_vec[start_index:end_index - 1].flatten()) # multiply according to the count of the year before
            ytp = ytp * np.sqrt(word_count_vec[start_index:end_index].flatten())
            

        yty_diff_makers = wc.wordWeightArrayTopN(yty_summands, word_dict, window_dict, top_n=10, signs=yty_signs)
        ytp_diff_makers = wc.wordWeightArrayTopN(ytp_summands, word_dict, window_dict, top_n=10, signs=ytp_signs)

        differences[word] = {"yty": yty, "ytp": ytp, "yty_diff_makers": yty_diff_makers, "ytp_diff_makers": ytp_diff_makers}
    
    return differences


"""
String Dict(String: Int) Dict(String <> Int) Dict(Any <> Int) -> Dict(Any: List(String))
Returns a dictionary that gives Dict[window_name] = [list of top ten words to occur in the context of word in the window]
"""
def getTopNContextWords(word, word_context_distrib, focal_word_dict, word_dict, window_dict):
    word_index = focal_word_dict[word]
    num_windows = len(window_dict) // 2
    start_index = word_index*num_windows
    end_index = start_index + num_windows
    return wc.wordWeightArrayTopN(word_context_distrib[start_index:end_index], word_dict, window_dict)


"""
Arr Bool Func Func -> Arr, Arr
Calculates a difference (based on distance metric) between distributions in Arr, depending on comp_func (which might compare to the last row,
or each row to each successive row). Also returns contribution of each column
"""
def time_series_differencing(arr, nonzeroify=False, comp_func=dist.make_shift_row_arrs, distance_metric=dist.calc_hellinger, var_arr=None):
    if nonzeroify:
        arr = nonzeroify(arr)
        
    arr_p, arr_q = comp_func(arr)

    if var_arr is None:
        return distance_metric(arr_p, arr_q)
    else:
        arr_p_var, arr_q_var = comp_func(var_arr)
        return distance_metric(arr_p, arr_p_var, arr_q, arr_q_var)

"""
Arr Float -> Arr
transforms a distribution with zero values into one with no zero values by incrementing all values by [epsilon] and dividing by the new sum
"""
def nonzeroify(arr, epsilon=0.0000001):
    arr = arr + np.ones(arr.shape)*epsilon
    arr = arr / arr.sum(axis=-1, keepdims=True)
    
    return arr
"""
Dict(String: Int) or List(String) Array Dict(String <> Int) -> Int String
Looks at all word-windows -- that is, occurrences in window i of word j (counted in wordcount_window_arr[i, j])
and finds the word-window with the lowest number of occurences. Also, if you set a min_threshold, it will remove
all words that don't always meet that min_threshold.

Returns the count of that word-window, the word it corresponds to, and the window index of that word-window
"""
def getLeastCommonWordWindow(wordcount_window_arr, word_dict, words_of_interest=None, start_ind=None, end_ind=None, min_threshold=None):
    min_word_count = float("inf")

    if words_of_interest is not None:
        word_indices = [word_dict[word] for word in words_of_interest]
        relevant_wordcount_window_arr = wordcount_window_arr[:, word_indices]
    elif start_ind is not None:
        relevant_wordcount_window_arr = wordcount_window_arr[:,start_ind:end_ind]

    if min_threshold is not None:
        meet_threshold = (relevant_wordcount_window_arr > min_threshold).all(axis=0)
        # print(meet_threshold.shape)
        indices_to_remove = np.nonzero(~meet_threshold)[0]
        # don't count the words not meeting threshold towards the min bar
        relevant_wordcount_window_arr = np.where(meet_threshold, relevant_wordcount_window_arr, float('inf'))
    else:
        indices_to_remove = []

    # print(relevant_wordcount_window_arr)
    least_common_flatten_ind = np.argmin(relevant_wordcount_window_arr) # gives index of the least common word

    # print(least_common_flatten_ind)
    number_of_words = relevant_wordcount_window_arr.shape[1]

    least_common_word_ind = least_common_flatten_ind % number_of_words
    least_common_wind_ind = least_common_flatten_ind // number_of_words
    # print(least_common_wind_ind, least_common_word_ind)

    min_count = relevant_wordcount_window_arr[least_common_wind_ind, least_common_word_ind]

    return min_count, least_common_wind_ind, indices_to_remove



"""
Int Any Int Series Dict(Int: Any) -> Any Int
Checks if [curr_window] matches whatever window corresponds to row i, if not, it increments
curr_window to the value of the window of row i, increments the window_counter, and matches the
index of the new window_counter to name of the new_window

def handleWindowCounter(i, curr_window, window_counter, window_col, window_dict):
    
    if curr_window != (new_window := window_col.iloc[i]):
        curr_window = new_window
        window_counter += 1
        window_dict[window_counter] = new_window
    
    return curr_window, window_counter
"""

"""
Arr[n, k, m] Arr[n, m] -> Arr[n, k, m]
Each 'final dimension' of the distribs are context distributions: specialDistribs[i, l, j] is the probability of seeing word j in scenario l
(usually in the context of word l) in window i. Essentially the "situation" is specified in 2d by the first two dimensions. In most cases,
the window and the word. We are seeing how the *second* dimension affects the representation of the word. so if the dimensins are window, focal_word,
context_word, we want to say, GIVEN a particular window, how much is a context word over- or under-represented in a particular context.

cutoff allows us to discount words that have a probability of essentially 0, because if they happen to appear in situation i, then
they will be extremely over-represented. any word that has P(j) < cutoff will have a value of 1 in the returned array.
"""
def calcOverUnderRepresentation(specialDistribs, generalDistrib, context_dict=None, word_dict=None, cutoff=0):
    if context_dict is not None:
        words_in_context_arr = [word for word in context_dict if isinstance(word, str)]
        words_in_context_arr_sorted = sorted(words_in_context_arr, key= lambda word: context_dict[word]) # sort by their index
        relevant_indices = [word_dict[word] for word in words_in_context_arr_sorted]

        generalDistrib = generalDistrib[:,relevant_indices]  
    
    num_windows = generalDistrib.shape[0]
    num_words = generalDistrib.shape[1]
    
    generalDistrib = np.reshape(generalDistrib, (num_windows, 1, num_words))
    
    mask = generalDistrib >= cutoff
    generalDistribCutoff = generalDistrib * mask
    specialDistribCutoff = specialDistribs * mask
    specialDistribNonzero = nonzeroify(specialDistribCutoff)
    generalDistribNonzero = nonzeroify(generalDistribCutoff)

    assert generalDistribNonzero.shape[0] == specialDistribNonzero.shape[0] == num_windows
    assert generalDistribNonzero.shape[1] == 1
    assert generalDistribNonzero.shape[2] == specialDistribNonzero.shape[2] == num_words

    
    return specialDistribNonzero / generalDistribNonzero

def entropy_by_last_dim(distribs):
    print(distribs.shape)
    entropy = - (distribs * np.log2(distribs)).sum(axis= -1)
    print(entropy.shape)
    return entropy

"""
Calculates the entropy of a set of samples from distributions where samples from a single distribution are given along the last dimension
"""
def nsb_entropy_last_dim(counts):
    alphabet_size = counts.shape[-1]
    entropy_func = lambda counts: ndd.entropy(counts, k=alphabet_size)

    entropies = np.apply_along_axis(entropy_func, -1, counts)
    return entropies

"""
Arr[n1, n2, ... ] Int Int -> Arr[dim0, dim1, n2, n3, ...]
"Rolls" up an array by taking the first dimension and splitting it into dim0, dim1. So, index i in the first dimension becomes 
(i % dim0, i // dim0) it must be that dim0 * dim1 = n1

"""
def roll_unrolled_arr(arr, dim0, dim1):
    rolled_dim = (dim0, dim1) + arr.shape[1:]
    arr_rolled = np.zeros(rolled_dim, dtype=arr.dtype)
    row_indices = np.array(range(arr.shape[0])) % dim0
    col_indices = np.array(range(arr.shape[0])) // dim0

    arr_rolled[row_indices,col_indices,...] = arr

    return arr_rolled
"""
*tested* but only for the case where context_pcts is already rolled!
"""
def compile_list_of_changes(context_pcts, num_words, comp_func=dist.make_shift_row_arrs, distance_metric=dist.calc_taxicab, context_vars=None):
    
    if len(context_pcts.shape) == 2:
        contexts_rolled = roll_unrolled_arr(context_pcts, -1, num_words)
        variances_rolled = roll_unrolled_arr(context_vars, -1, num_words) if context_vars is not None else None
    else:
        contexts_rolled = context_pcts
        variances_rolled = context_vars
        
    word_changes, word_change_summands, word_change_signs = time_series_differencing(contexts_rolled, nonzeroify=False, comp_func=comp_func, distance_metric=distance_metric, var_arr=variances_rolled)

    return word_changes, word_change_summands

def compile_list_of_frequencies(word_window_counts, focal_word_dict, word_dict, num_windows, normalize=True):
    num_words = len(focal_word_dict) // 2
    
    if normalize:
        word_window_counts = word_window_counts / word_window_counts.sum(axis=1, keepdims=True)

    orig_word_indices = [word_dict[word] for word in focal_word_dict if isinstance(word, str)]
    word_col_indices = np.array([orig_word_indices])
    wind_row_indices = np.array(range(num_windows)).reshape(num_windows, 1) # repeat word_index
    
    print(wind_row_indices)
    print(word_col_indices)

    word_window_counts = word_window_counts[wind_row_indices, word_col_indices]
    # word_window_counts_truncated = word_window_counts[0:num_windows - 1, ...]

    return word_window_counts

    # for i in range(num_windows):
 
def do_regression(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    
    x = sm.add_constant(x)

    olsmod = sm.OLS(y, x)
    olsres = olsmod.fit()
    print(olsres.summary())

    y_pred = olsres.predict(x)
    
    return olsres, y_pred
"""
List(List(String)) Series Dict(String <> Int) Int Int Int -> Arr(top_n, top_m)
Generates an array shaped (top_n, top_m), where each row corresponds to a focal word and each column is the probability
that a given word is in the focal word's context (a window of [frame] words forward and back). That is Arr[i, j] is the 
probability that a given word in the context of word i is word j.

Requires a word dictionary where the words are indexed according to their frequency

Takes the top_n words as focal words and the top_m words as slots in the distribution.

def genWordContextDistribs(list_of_speech_lists, window_col, sorted_word_dict, frame=10, top_n=5000, top_m=30000):
    
    word_context_distrib = np.zeros((top_n, top_m))
    
    for speech in list_of_speech_lists:
        current_context = [sorted_word_dict[speech[i]] for i in range(frame + 1) 
                           if i < len(speech)]
        current_context = [item if item < top_m else None for item in current_context]
        context_ind = 0
        
        for i, focal_word in enumerate(speech):
            collapsed_context = [item for item in current_context if item is not None]
            if (focal_ind := sorted_word_dict[focal_word]) < top_n:
                word_context_distrib[focal_ind, collapsed_context[:context_ind]] += 1
                word_context_distrib[focal_ind, collapsed_context[context_ind + 1:]] += 1
            
            # print(focal_ind)
            # print(context_ind)
            # print(current_context)
            
            # if we are not smushed against the beginning of the list
            if context_ind == frame:
                current_context = current_context[1:]
            else:
                context_ind += 1
            
            # if we are not smushed against the end of the list
            if i + 1 + frame < len(speech):
                potential_ind = sorted_word_dict[speech[i + 1 + frame]]
                
                if potential_ind < top_m:
                    current_context.append(potential_ind)
                else:
                    current_context.append(None)
                       
    word_context_distrib = word_context_distrib / word_context_distrib.sum(axis=1, keepdims=True)
    return word_context_distrib
"""