# word-evolution

## the `share_data` directory 

`full_2tuple_arr_*.pkl` are coordinate arrays as implemented in the `sparse` library (`pip install sparse`), counting the number of cooccurences of pairs in each window. The pairs counted are those that have one word in the top 500 and both words in the top 1000. If you load the pickle file and call it `arr`, then `arr.coords` gives a `numpy` array of shape `(2, n)` where `n` is the number of pairs. Each column in this array is a pair. Then, the corresponding index in `arr.data` gives the counts of that pair.

`full_word_dict.pkl` is a dictionary translating between the word index and the actual word. 

`full_wordcount_window.pkl` is an array of shape `(14, n)` where `n` is the number of words, and where the [i, j] index is the number of occurrences of word `j` in window `i`. 

