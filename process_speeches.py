import re
from tqdm import tqdm
import add_metadata as am
import word_counting as wc
import numpy as np

import multiprocessing
import pandas as pd
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

tqdm.pandas()

from nltk.util import ngrams

def generate_random_speeches(winds, word_dist, wordpspeech, speechpwind):
    # word_dist = [6/20, 5/20, 4/20, 3/20, 2/20]
    speeches = {
        wind: [np.random.choice(range(len(word_dist)), (wordpspeech,), p=word_dist) for j in range(speechpwind)] 
            for wind in range(winds)
    }
    
    return speeches

def generate_sample_data(manyWindows=False):
    data_dict = {
        "chamber": ["S", "H", "S", "S", "S", "H", 
                    "H", "H", "S", "H", "S", "S", 
                    "H", "H", "H", "S", "H", "S",
                    "S", "S", "S", "H"],
        "date": [20170805, 20171202, 20171227, 20180213, 20181104, 20181216, 20181230, 20190303, 20190919, 20191130, 20191225, 
                 20200408, 20200509, 20201013, 20210107, 20210809, 20211119, 20211230, 20220329, 20221120, 20221225, 20221230],
        "speaker": ["Victor Odouard", "Eddie Lee", "David Krakauer", "Melanie Mitchell", "Cris Moore", "David Wolpert", "Geoffrey West", "Fred Cooper", "Paul Krapivsky", "Francois Odouard", "Ilina Odouard",
                    "Sid Redner", "Jonas Dalege", "Andres Ortiz", "Helena Miton", "Mirta Galesic", "Paulina Bitternova", "Brigitte Odouard", "Tiphaine Burguburu", "Kovid Odouard", "", "Your mom"],
       "speech": ["She looked out at the stunning blue sea from her balcony in Paris. Her girlfriend would be arriving tomorrow, and they planned to spend their vacation exploring the city together.",
                  "Her stunning blue eyes looked almost transparent in the morning light. We had planned to spend tomorrow working, but that moment I couldn't resist asking her to take a day trip with me.",
                  "The feeling of loss when he left the job was visceral, but eventually he learned to cope with it. As he walked home, he clutched his notebook and bag tightly, as if to hold on to the one relic from his last few years of toil.",
                  "He absentmindedly twirled his bag in the stunning blue light, by the sea. He was lost in thought on the street, which gradually emptied out as rush hour ended, people settling into their desks.",
                  "He and his wife had a difficult conversation by the fireplace, but afterward, their relationship felt stronger than ever. While they disagreed in their politics, he felt grateful that they were so similar in other respects.",
                  "The children watching the tree grow were between the ages of four and about seven, but they had an earnestness about them that rose well beyond their years. Someone told these children to play in the park, but they were more interested in caring for their garden.",
                  "His wife was giving birth, and the doctor anxiously clutched his notebook and bag as he realized something was not quite right. This didn't exactly reassure him.",
                  "The child sat on the steps of the school, waiting for her mother to pick her up. She clutched her notebook and bag tightly, eager to show her what she had learned.",    
                  "He drove his car up the winding mountain road, enjoying the crisp air and breathtaking view of the trees below. He couldn't wait to hike to the top and take in the panoramic scenery.",
                  "As I was arriving at the hotel, tired and jet-lagged, I looked up to see my girlfriend standing on the balcony. She looked stunning, with the city skyline as her backdrop. I couldn't wait to join her, to hold her in my arms and breathe in the new energy of the city.",
                  "Arriving in Paris to meet my girlfriend felt like a dream come true. She was waiting for me at the train station, her face lighting up as we locked eyes. Together, we took in the beauty of the city, from the glittering Eiffel Tower to the winding streets filled with charming cafes and shops.",
                  "The woman walked down the quiet street, lost in thought. She had just left her job and was considering starting her own business. She absentmindedly twirled her bag as she brainstormed ideas.",    
                  "She sat by the fireplace, engrossed in conversation with her friends. They talked about everything from politics to travel, and she felt grateful for their support and encouragement.",    
                  "He waited anxiously in the hospital lobby, clutching his bag and pacing back and forth. His wife was giving birth to their first child, and he couldn't wait to hold his newborn son.",    
                  "She walked down the cobblestone streets of Paris, marveling at the beauty and history of the city. She stopped to rest on a bench under a tree, watching children play in the park.",    
                  "He sat in his car, waiting for the rain to stop. He glanced at his notebook, where he had written down his to-do list for the day. He hoped to make it to the post office before it closed.",    
                  "The child stood in awe at the foot of the mountain, staring up at its imposing peak. She clutched her mother's hand tightly, excited for their hike to the top.",    
                  "She sat on the steps of the school, talking to her friends about their plans for the weekend. They were going to take a road trip in her car and explore a nearby mountain town.",    
                  "He sat by the fireplace, lost in thought. He had a big talk with his boss tomorrow, where he would ask for a promotion. He rehearsed his arguments in his head as he watched the flames dance.",    
                  "She walked down the bustling city street, dodging cars and people left and right. She was late for her meeting at the school, where she was pitching a new program to the administration. She hoped she wouldn't be too out of breath when she arrived.",
                  "The quiet woman walked down the street, clutching her bag. Back in her youth, her mom told told her that she would be a famous movie star, but that didn't happen. She felt grateful for the encouragement at the time, but now, she thought, it made her feel lost. The street thought otherwise.",
                  "He had left his job, and it made him feel a deep-seated anxiety. He wasn't sure how he would pay the electric bill that he would receive tomorrow. He was considering a new job, and might have to take it if he wasn't going to declare bankruptcy."
                  ]

    }

    speeches = pd.DataFrame(data_dict)
    speeches["year"] = am.make_year_column(speeches)
    speeches["wind"] = am.make_n_year_groupings(speeches, n=1) if manyWindows else 1

    speech_list_dict = make_dict_of_speech_lists(speeches)

    wordcount_window, word_dict, window_dict = wc.get_wordcount_arr(speech_list_dict, 
                                                                   word_dict=None,
                                                                   top_m=50000)

    return speeches, speech_list_dict, wordcount_window, word_dict, window_dict

"""
String -> String
removes punctuation, newlines, and back-to-back spaces from <text>, also makes
everything lowercase
"""
def remove_punc_and_lower(text):
    text = text.replace("\n", " ")
    text = re.sub('[^\w\s]', '', text)
    text = re.sub('[\s\s]+', " ", text)
    text = text.lower()
    
    # p = PorterStemmer()
    # text = p.stem(text)
    
    return text

"""
List(List) List(String) -> List(List)
removes [stop_words] from every speech in [list_of_speech_lists] 
"""
def remove_stopwords(list_of_speech_lists, stop_words):
    return [[word for word in speech_list if word not in stop_words] for speech_list in list_of_speech_lists]

"""
pd.Series -> 

calls [remove_punc_and_lower] on each speech in a series of speeches (col of df)
and produces a new column with the processed version of the speech
"""
def make_processed_speech_column(speeches):
    new_col = speeches["speech"].progress_map(remove_punc_and_lower)
    return new_col

"""
String Bool Bool -> List(String)
does a simple preprocessing step on [speech], which includes removing punctuation and special characters, and separating
the text into a list of tokens (tokenization). If filter stopwords is true, it also removes stopwords
"""
def full_preprocess(speech, filter_stopwords=True, deacc=True):
    speech = simple_preprocess(speech, deacc=True, min_len=1, max_len=30)
    
    if filter_stopwords:
        stop_words = stopwords.words("english")
        stop_words.extend(["mr", "bill", "would", "chairman", "gentleman", "amendment", "time", "committee", "speaker", "xz"])
    else:
        stop_words = []

    speech = [word for word in speech if word not in stop_words]
    
    return speech

"""
**tested**
"""
def convert_words_to_ints(speech_dict_list, word_dict, window_dict):
    new_speech_list_dict = {}
    for wind, speech_list in speech_dict_list.items():
        new_speeches_list = []
        for speech in speech_list:
            new_speech = np.array([word_dict[word] for word in speech if word in word_dict], dtype=np.uint16)
            new_speeches_list.append(new_speech)
        new_wind = window_dict[wind]
        new_speech_list_dict[new_wind] = new_speeches_list

    return new_speech_list_dict


def tokenizeAllSpeechesByWindow(speeches, filter_stopwords=True, parallel=True):
    windows = speeches["wind"].unique()
    windowDfs = {wind: speeches[speeches.wind == wind].speech for wind in windows}
    inputs = [(df, filter_stopwords) for df in windowDfs.values()]

    if parallel:
        with multiprocessing.Pool() as pool:
            list_of_speech_lists = pool.starmap(makeListOfSpeechListsWind, inputs)
    else:
        list_of_speech_lists = list(map(makeListOfSpeechListsWind, inputs))

    dict_of_speech_lists = dict(zip(windowDfs.keys(), list_of_speech_lists))

    return dict_of_speech_lists

"""
Series -> List(List(String))
takes a Seris of speeches (df column, often) and makes each one into a list of tokens
**tested**
"""
def make_dict_of_speech_lists(speeches, filter_stopwords=True, parallel=True):
    return tokenizeAllSpeechesByWindow(speeches, filter_stopwords, parallel)

def makeListOfSpeechLists(dictOfSpeechLists):
    list_of_speech_lists = []
    [list_of_speech_lists := list_of_speech_lists + lst for lst in dictOfSpeechLists.values()]
    return list_of_speech_lists

        

def makeListOfSpeechListsWind(speeches, filter_stopwords=True):
    list_of_speech_lists = [full_preprocess(speech, filter_stopwords, deacc=True) for speech in speeches.values]

    return list_of_speech_lists

"""
List(List(String)) Int -> List(List(String))
Makes the ngram version of the list of speech lists
"""
def make_list_of_speech_lists_ngram(list_of_speech_lists, n):
    return [list(ngrams(speech_list, n)) for speech_list in list_of_speech_lists]  

"""
Series Series -> List(List(Strings))
Makes a list of token lists, where a token list is no longer a speech, but a grouping as determined by the cat_column
"""
def make_list_of_window_lists(speeches, cat_column):
    cat_ids = speeches[cat_column].unique()
    concat = lambda x, y: x + y
    
    list_of_debate_lists = []
    
    for cat_id in cat_ids:
        list_of_speech_lists = make_list_of_speech_lists(speeches[speeches[cat_column] == cat_id])
        debate = []; [debate := debate + speech for speech in list_of_speech_lists]
        list_of_debate_lists.append(debate)
    
    # print(len(cat_ids))
    # print(len(list_of_debate_lists))
    
    return list_of_debate_lists

"""
DataFrame, Any, List(List(String)), Series-> DataFrame, List(List(String))
Removes all eleements from the DataFrame that are in grouping [remove_value],
and also removes the corresponding elements from [speech_list]
"""
def remove_grouping(speeches, remove_values, grouping_col, speech_list=None):
    if "ind" in speeches:
        speeches.loc[:,"ind"] = range(len(speeches))
    else:
        speeches["ind"] = range(len(speeches))
    
    mask = (~grouping_col.isin(remove_values))
    new_speeches = speeches[mask]

    if speech_list is not None:
        kept_indices = speeches.ind[mask]
        new_speech_list = [speech_list[i] for i in kept_indices.values]
        
        return new_speeches, new_speech_list
    
    return new_speeches
