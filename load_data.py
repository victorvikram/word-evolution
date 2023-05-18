import pandas as pd 
import random
import re
import os, glob
import sys

import pickle as pkl



"""
reads the stanford daily dataset into a dataframe with speech and speech metadata
"""
def read_hein_text_files_to_df(data_dirs, max_size=float("inf"), min_word_count=None, skip=1, prob=1, check_repeats=True):

    speech_file_list = []
    speech_file_basenames = []

    for directory in data_dirs:
        speech_filename_template = os.path.join(directory, "speeches_*.txt")

        for f in glob.glob(speech_filename_template):
            f_base = os.path.basename(f) 
            if not check_repeats or f_base not in speech_file_basenames:
                speech_file_list.append(f)
                speech_file_basenames.append(f_base)
      
    dfs = []
    
    print(speech_file_list)
    counter = 0
    for speech_file in speech_file_list:
        speech_dir = os.path.dirname(speech_file)
        
        speech_number = re.split("_|\.", os.path.basename(speech_file))[1]
        speaker_file = os.path.join(speech_dir, f"{speech_number}_SpeakerMap.txt")
        description_file = os.path.join(speech_dir, f"descr_{speech_number}.txt")
        print(speech_file, speaker_file, description_file)

        df_speeches = pd.read_csv(speech_file, delimiter="|", on_bad_lines='warn', encoding_errors='replace', engine='python')
        df_description = pd.read_csv(description_file, delimiter="|", on_bad_lines='warn', encoding_errors='replace', engine='python')
        df_speaker = pd.read_csv(speaker_file, delimiter="|", on_bad_lines='warn', encoding_errors='replace', engine='python')

        # skip speeches and take sample
        df_speeches = df_speeches[::skip]
        df_speeches = df_speeches.sample(frac=prob)

        df_speeches = df_speeches.set_index("speech_id")
        df_description = df_description.set_index("speech_id")
        df_speaker = df_speaker.set_index("speech_id")
        
        df_speeches = df_speeches.join(df_speaker["speakerid"], how="left")
        
        df_speeches = df_speeches.join(df_description.loc[:,["chamber","date","speaker","first_name","last_name","state","gender","word_count"]], how="left")
        
        if min_word_count is not None:
            df_speeches = df_speeches[df_speeches.word_count > min_word_count]
        
        counter += len(df_speeches)
        dfs.append(df_speeches)
    
        if counter > max_size:
            break
    
    print(len(dfs))
    speeches = pd.concat(dfs)
    speeches["speech"] = speeches["speech"].fillna("")
    
    return speeches

def dump_one_by_one(dct, name, parent="objects", window_dict=None):

    for key, item in dct.items():
        if window_dict is not None:
            key = window_dict[key]
        
        with open(os.path.join(parent, f"{name}_{key}.pkl"), "wb") as f:
            pkl.dump(item, f)

def read_by_key(name, key, parent="objects"):
    with open(os.path.join(parent, f"{name}_{key}.pkl"), "rb") as f:
        val = pkl.load(f)
    
    return val

def get_keys_for_name(parent, name, dtype=lambda x: x):
    file_list = glob.glob(os.path.join(parent, f'{name}_*.pkl'))

    keys = []
    # extract the key from the filename
    for fl in file_list:
        match = re.search('_(\d+)\.pkl', fl)
        key = match.group(1)
        print(key)
        keys.append(dtype(key))
    
    return keys



def read_one_by_one(name, dtype=lambda x: x):
    full_dct = {}

    keys = get_keys_for_name(name)
    for key in keys:
        val = read_by_key(name, key)
        full_dct[key] = val

    return full_dct

"""
creates a dataframe with convote speeches and metadata
"""
def read_convote_text_files_to_df(txt_file_dir, max_size=None):
    filename_template = os.path.join(data_dir, "*.txt")
    file_list = [i for i in glob.glob(filename_template)]
    
    column_names = list(parse_convote_filename().keys())
    column_names.append("speech")
    
    df_length = len(file_list) if max_size is None else max_size
    
    df = pd.DataFrame(index=range(df_length),columns=column_names)
    for i, filename in enumerate(file_list[:df_length]):
        with open(filename, 'r') as f:
            data = f.read()
        
        columns = parse_convote_filename(filename)
        columns["speech"] = data
        
        df.loc[i, columns.keys()] = list(columns.values())
            
    return df

"""
returns a dictionary with the metadata of a particular speech
"""
def parse_convote_filename(filename=""):
    if filename != "":
        base_name = os.path.basename(filename)
        segments = base_name.split("_")
        bill_id = segments[0]
        speaker_id = segments[1]
        page = segments[2][:4]
        index = segments[2][4:]
        party = segments[3][0]
        mention = segments[3][1]
        vote = segments[3][2]
    else:
        bill_id = speaker_id = page = index = party = mention = vote = None
    
    return {"bill_id": bill_id, "speaker_id": speaker_id, "page": page, "index": index, "party": party, "mention": mention, "vote": vote}


def get_full_size(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_full_size(x) for x in obj)
    elif isinstance(obj, dict):
        size += sum(get_full_size(k) + get_full_size(v) for k, v in obj.items())
    return size

def dump_counts(wordcount_window, word_dict, window_dict, pref, parent="objects"):
    with open(os.path.join(parent, f"{pref}_wordcount_window.pkl"), "wb") as f:
        pkl.dump(wordcount_window, f)
    with open(os.path.join(parent, f"{pref}_word_dict.pkl"), "wb") as f:
        pkl.dump(word_dict, f)
    with open(os.path.join(parent, f"{pref}_window_dict.pkl"), "wb") as f:
        pkl.dump(window_dict, f)

def load_counts(cat):
    with open(os.path.join("objects", cat, f"{cat}_wordcount_window.pkl"), "rb") as f:
        wordcount_window = pkl.load(f)
    with open(os.path.join("objects", cat, f"{cat}_word_dict.pkl"), "rb") as f:
        word_dict = pkl.load(f)
    with open(os.path.join("objects", cat, f"{cat}_window_dict.pkl"), "rb") as f:
        window_dict = pkl.load(f)
    
    return wordcount_window, word_dict, window_dict
    