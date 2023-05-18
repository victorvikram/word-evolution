import networkx as nx
import numpy as np

def safe_reciprocal(arr):
    numerator = np.where(arr != 0, 1, 0)
    denominator = np.where(arr != 0, arr, 1)

    return numerator / denominator

def safe_complement(arr, epsilon=0.00001):
    subtractor = np.where(arr != 1, arr, 1 - epsilon)
    return 1 - arr

def betweenness_centrality(adj_mat, edge_adjustment=safe_reciprocal):
    adj_mat = edge_adjustment(adj_mat)
    G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)

    print(nx.number_strongly_connected_components(G))
    
    return nx.betweenness_centrality(G)

def pagerank_centrality(adj_mat):
    G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
    print(nx.number_strongly_connected_components(G))
    return nx.pagerank(G)

def matrixify_dctlst(dctlst):
    lst_of_lsts = [[val for key, val in dct.items()] for dct in dctlst]
    mat = np.array(lst_of_lsts)
    return mat

"""
Arr[num_windows,num_words,num_words] -> Arr[num_windows, num_words]
Each slice of the Arr is a relational network among words for a particular time window. The function calculates the centrality of 
each word (Arr[i, j, k] is the link going from j to k in window i) and returns a matrix where centrality_mat[i, j] is the centrality
of word j in window i.
"""
def centrality_by_layer(rolled_mat):
    centrality_dicts = []

    for i in range(rolled_mat.shape[0]):
        print(i)
        flat_adj_mat = rolled_mat[i,:,:]
        centrality_dict = pagerank_centrality(flat_adj_mat)
        centrality_dicts.append(centrality_dict)
    
    centrality_mat = matrixify_dctlst(centrality_dicts)

    return centrality_mat



