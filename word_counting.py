import numpy as np
import multiprocessing
from tqdm import tqdm

"""
Arr[n, m] Dict(String <-> Int) Int -> Arr[n, m or truncate_at] Dict(String <-> Int)
Takes an array of word counts, which is Arr[i, j] = the count of word j in window i (the window could be a speech, or a year,
or some other grouping). Sorts the columns in descending order of word frequency across all windows, and returns a new
two-sided dictionary with the correspondence between words and indices in the new Arr
**tested**
"""
def sort_wordcount_arr(wordcount_arr, word_dict, truncate_at=None):
    
    if truncate_at is None:
        truncate_at = wordcount_arr.shape[1]
    
    word_totals = np.sum(wordcount_arr, axis=0)
    indices = np.argsort(-word_totals)
    indices = indices[:truncate_at]
    wordcount_arr_sorted = wordcount_arr[:,indices]
    
    word_dict_sorted = {i: word_dict[ind] for i, ind in enumerate(indices)}
    word_dict_sorted.update({word_dict[ind]: i for i, ind in enumerate(indices)})
        
    return wordcount_arr_sorted, word_dict_sorted

"""
Arr(n,) Int -> Arr(n or 2n,)
If the counter is out of range of the length of the array, it doubles the length of the array
"""
def growArrayIfNecessary(arr, counter):
    length = arr.shape[0]
    
    if counter >= length:
        new_arr = np.zeros((length,))
        arr = np.append(arr, new_arr)
        
    return arr



"""
List(List(String)) Int -> Dict(String <> Int)
Makes a dictionary of the top_m words, taken from a sample of all the speeches. The dictionary is sorted based on
how often the word appears in the sample. If top_m is none, doesn't truncate the list at all. 
Returns the the word_dict, and the count of words initially discovered before truncation
"""
def makeWordDict(speech_list_dict, top_m=None):
    speech_list_full = []
    [speech_list_full := speech_list_full + speech_list for speech_list in speech_list_dict.values()]

    desired_sample = min(100000, len(speech_list_full))
    step_size = len(speech_list_full) // int(desired_sample)
    
    word_count = np.zeros((desired_sample,)) # approx one new word per speech
    counter = 0
    
    word_dict = {}
    
    for speech in tqdm(speech_list_full[::step_size]):
        for word in speech:
            if word not in word_dict:
                word_dict[word] = counter
                word_dict[counter] = word
                counter += 1
                word_count = growArrayIfNecessary(word_count, counter)
            
            word_count[word_dict[word]] += 1
    
    indices = np.argsort(-word_count)[:top_m]
    word_dict_pruned = {i: word_dict[index] for i, index in enumerate(indices) if index in word_dict}
    word_dict_pruned.update({word_dict[index]: i for i, index in enumerate(indices) if index in word_dict})
    
    return word_dict_pruned, counter
    

def dict_lookup_inverse(dct, val):
    keys = [key for key, cand in dct.items() if cand == val]
    return keys         


"""
Arr Arr -> Arr
Takes an array of word occurences by speeech and by grouping and calculates a score which is higher if the word is less frequent,
but also higher if the word shows up more consistently in the grouping
"""
def get_less_frequent_more_consistent_words_score(wordpresence_pct_spch, wordpresence_pct_dbt):
    word_score = (1 - wordpresence_pct_spch)**2 * wordpresence_pct_dbt
    return word_score



# List(List(String)) Series Dict Bool -> Array Array Dict
# from a list of "speech lists", which are themselves lists of tokens that occured in a particular speech
# this function returns an n by m array, where m is the number of words, and n is the number of speeches.
# wordcount_speech_arr[i, j] is the number of occurences of word j in speech i
# wordcount_window_arr[i, j] is the number of occurences of word j in window i
# the word_dict is a bidirectional dictionary: Dict[word] = i, the index of that word, and Dict[i] = word, 
# the word corresponding to the index.
# if speech_counter is False, doesn't make the wordcount_speech_counter
# if top_m is not none, then we only count the top_m words as determined by a random sample of speeches
# **tested**
def get_wordcount_arr(speech_list_dict, word_dict=None, top_m=None, parallel=False, already_numbers=False):
    if word_dict is None and not already_numbers:
        word_dict, _ = makeWordDict(speech_list_dict, top_m=top_m)
    
    if not already_numbers:
        num_words = len(word_dict) // 2
    else:
        num_words = max([speech.max() + 1 for speech_list in speech_list_dict.values() for speech in speech_list]) # need the array to be one bigger than the max element

    speech_lists_ext = [(speech_list, word_dict, wind, num_words, already_numbers) for wind, speech_list in speech_list_dict.items()]
    window_dict = {wind: i for i, wind in enumerate(sorted(speech_list_dict.keys()))}
    window_dict.update({wind: i for i, wind in enumerate(sorted(speech_list_dict.keys()))})

    if parallel:
        with multiprocessing.Pool() as pool:
            wordcount_window_lst = pool.starmap(get_wordcount_arr_window, speech_lists_ext)
    else:
        wordcount_window_lst = list(map(lambda args: get_wordcount_arr_window(*args), speech_lists_ext))
    
    wordcount_window_arr = np.concatenate(wordcount_window_lst)

    return wordcount_window_arr, word_dict, window_dict

# takes a speech list and gets the word count in the speech list for all the words in word_dict (`wind` is just for hte progress bar)
# **tested**
def get_wordcount_arr_window(speech_list, word_dict, wind, num_words, already_numbers=False):
    def get_index(word):
        if already_numbers:
            return word
        elif word in word_dict:
            return word_dict[word]
        else:
            return None
    
    print(f"starting {wind}", end="\r")
    wordcount_window_arr = np.zeros((1, num_words))
    for speech in speech_list:
        for word in speech:
            ind = get_index(word)
            if ind is not None:
                wordcount_window_arr[0,ind] += 1
    
    print(f"ending {wind}", end="\r")
    return wordcount_window_arr


# Arr -> Vec
# Arr is a table of word counts for given groupings, could be a speech or a debate or a bill. 
# Arr[i, j] is the number of occurrences of word j in grouping i
# Vec[i] is the percentage of groupings in which word i shows up.
def get_wordpresence_pct(wordcount_speech_arr):
    wordpresence_pct = (wordcount_speech_arr > 0).astype(int).sum(axis=0) / wordcount_speech_arr.shape[0]
    # wordpresence_variance = wordpresence_pct * (1 - wordpresence_pct) # this is proportional to the variance np(1 - p)
    # wordpresence_variance_norm = wordpresence_variance / wordpresence_pct [this is the best measure so far]
    # print_word_quantity_vector(word_dict, wordpresence_variance_norm) 
    # print_word_quantity_vector(word_dict, wordpresence_variance_norm)
    
    return wordpresence_pct


# TODO possibly fix the following functions because now the wordcount array is the transpose of what it used to be

# Arr[m, n] -> Arr[m, m]
# the input array is the number of occurrences of a word per grouping. the output array is the correlation coefficient
# between word i and word j. It seems to me that the more frequent words just have higher correlations
def get_word_correlations(wordcount_speech_arr):
    # correlation words -- which words have intense "regulatory" effects on others?
    # will this be biased to more common words? it is.
    word_corrs = np.corrcoef(np.transpose(wordcount_speech_arr))
    word_corr_strengths = np.absolute(word_corrs).sum(axis=1)
    #print_word_quantity_vector(word_dict, word_corr_strengths)
    
    return word_corrs

# Arr[m, n] Arr[m, n] Int Int -> Arr[m, top_n]
# Returns the column indices of the top n items from each row of val_arr whose corresponding entry in thresh_arr meets the threshold
# note: to get the lowest values that meet the threshold, simply negate val_arr. To get the highest values that
# are under the threshold, simply negate the threshold and the thresh_arr. Any thresh entry that is now above the new threshold
# was below the original threshold
def get_highest_meeting_threshold(val_arr, thresh_arr, threshold, top_n=50):
    val_arr = np.where(thresh_arr > threshold, val_arr, float('-inf')) # zero out entries where the thresh_arr entry doesn't meet threshold
    print(val_arr)
    highest_indices = np.argsort(-val_arr, axis=1)[:,:top_n]
    print(highest_indices)
    
    return highest_indices

    
# get either the 50 rarest words that nonetheless show up in over [consistency_threshold] fraction of debates
# or all the words that show up in over [consistency_threshold] fraction of debates 
# returns a list of words sorted by rarity (most rare first)
def get_less_frequent_more_consistent_words(wordpresence_pct_spch, 
                                            wordpresence_pct_dbt,
                                            word_dict,
                                            consistency_threshold):
    
    sorted_indices = wordpresence_pct_spch.argsort()
    
    less_frequent_more_consistent_words = []
    counter = 0
    while len(less_frequent_more_consistent_words) < 50 and counter < len(sorted_indices):
        ind = sorted_indices[counter]
        
        # translate from spch_ind to dbt_ind
        word = word_dict[ind]
        
        if wordpresence_pct_dbt[ind] >= consistency_threshold:
            print("Word meeting criteria: ", word)
            less_frequent_more_consistent_words.append(word)
        else:
            print("Word skipped: ",  word)
        
        counter += 1
            
    return less_frequent_more_consistent_words

"""
Arr[n, m] Dict(String <> Int) Dict(String: Int) Int -> Dict(Any: List(String))
word_weights[i, j] is a "weighting" (could be interpreted as a probability, or some other indicator of strength)
of word j in window i.

Returns a dictionary where the keys are the window names and the values are the top_n values from that window
"""
def wordWeightArrayTopN(word_weights, word_dict, row_dict, top_n=10, signs=None):
    important_indices = np.argsort(-word_weights, axis=1)[:,:top_n]
    row_count = word_weights.shape[0]

    if signs is not None:
        row_inds = np.array(range(row_count)).reshape(row_count, 1)
        relevant_signs = signs[row_inds, important_indices]
    else:
        relevant_signs = np.ones((word_weights.shape[0], top_n))
    
    assert relevant_signs.shape[0] == row_count
    assert relevant_signs.shape[1] == top_n
    
    topn_by_row = {}

    def get_sign_string(sign):
        if sign == 1:
            return "+"
        elif sign == -1:
            return "-"
        elif sign == 0:
            return "0"
    
    for i, row in enumerate(important_indices):
        assert len(row) == len(relevant_signs[i,:])
        topn_by_row[row_dict[i]] = [(get_sign_string(sign), word_dict[context_word_ind]) for context_word_ind, sign in zip(row, relevant_signs[i,:])]
    
    return topn_by_row
    
"""
def make_word_dict(list_of_speech_lists):
    word_dict = {}
    counter = 0
    
    for i, speech_list in tqdm(enumerate(list_of_speech_lists)):
        for word in speech_list:
            if word not in word_dict:
                word_dict[word] = counter
                word_dict[counter] = word
                counter += 1
    
    return word_dict
"""