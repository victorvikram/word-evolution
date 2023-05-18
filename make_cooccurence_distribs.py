import context_distribs as cd
import word_counting as wc
import os
import sys
import pickle as pkl
import time
import display as d
import psutil 
import gc
from pympler import asizeof

import load_data as ld

"""
python make_cooccurence_distribs.py full
python make_cooccurence_distribs.py mini

"""

WINDOWS_TO_OMIT = []
GET_TUPLE_DICT_FROM_EXISTING_FILE = False
GET_TRIPLETS = True

if __name__ == "__main__":
    t0 = time.time()
    cat = "full" if len(sys.argv) < 2 else sys.argv[1]

    if GET_TUPLE_DICT_FROM_EXISTING_FILE:
        existing_file = os.path.join("objects", cat, f"{cat}_3tuple_dict_0.pkl")

    output_name_pair = os.path.join(os.path.join("objects", cat), f'{cat}_pair_dist')
    output_name_trip = os.path.join(os.path.join("objects", cat), f'{cat}_trip_dist')

    window_wordcount, word_dict, window_dict = ld.load_counts(cat)
    windows = ld.get_keys_for_name(os.path.join("objects", cat), f"{cat}_speech_dict", int)
    windows = [wind for wind in windows if wind not in WINDOWS_TO_OMIT]

    print(windows)

    if GET_TUPLE_DICT_FROM_EXISTING_FILE:
        with open(existing_file, "rb") as f:
            trip_dict = pkl.load(f)
            trip_dict = {key: 3 for key in trip_dict}
            pair_dict = {}
            cd.match_pair_trip_dict(pair_dict, trip_dict, num_focal_words=500, num_context_words=1000)
    else:
        pair_dict, trip_dict = cd.gen_pair_trip_dict(cat, windows, 1/100, frame=20, num_focal_words=500, num_context_words=1000, parallel=True, par_memo=False)

    print(len(trip_dict))
    # y = input("x")
    if GET_TRIPLETS:
        cd.genNWiseCooccurenceDistribs(cat, windows, 3, frame=20, num_focal_words=500, num_context_words=1000, parallel=True, par_memo=False, tuple_dict=trip_dict, tuple_thresh=2, return_dict=False)
    
    cd.genNWiseCooccurenceDistribs(cat, windows, 2, frame=20, num_focal_words=500, num_context_words=1000, parallel=True, par_memo=False, tuple_dict=pair_dict, return_dict=False)
    

