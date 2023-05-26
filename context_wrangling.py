import load_data as ld
import numpy as np

"""
returns an array of shape `(samples, frame)` containing a bunch of contexts of size `frame`
"""
def sample_contexts_wind(cat, wind, frame=20, samples=100000):
    speech_list = ld.read_by_key(f"{cat}_speech_dict", wind, parent=f"objects/{cat}")
    lengths = np.array([len(speech) if len(speech) >= frame else 0 for speech in speech_list])
    
    sample = np.random.choice(range(len(speech_list)), size=(samples,), p=lengths/np.sum(lengths))
    start_pct = np.random.rand(samples)
    start_point = np.floor(start_pct * (lengths[sample] - (frame - 1))).astype(int)
    
    selected_contexts = np.array([speech_list[i][start_point:start_point+frame] for i, start_point in zip(sample, start_point)])

    return selected_contexts
"""
this takes the context array and converts it to a binary array of shape `(contexts.shape[0], top_n + 1)`
where [i, j] is the number of appearances of word j in context i. The last context is the aggregate of all
words outside the top_n range
"""
def context_to_count(contexts, top_n=500):
    contexts = np.where(contexts < top_n, contexts, top_n)
    num_samples = contexts.shape[0]
    count_arr = np.zeros((num_samples, top_n + 1))

    np.add.at(count_arr, (np.arange(num_samples).reshape(-1, 1), contexts), 1)
    
    return count_arr