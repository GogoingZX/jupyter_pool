import operator
import numpy as np

def knn_classify(
    input_vector,
    trainig_set,
    labels,
    k
):
    """
    Here is the description
    """
    
    data_size = trainig_set.shape[0]
    diff_matrix = np.tile(input_vector, (data_size, 1)) - trainig_set # np.tile(input_array, repetitions)
    sq_diff_matrix = diff_matrix ** 2
    sq_distance = sq_diff_matrix.sum(axis=1)
        # >> sum(axis=1)
        # data1: (d1, d2, d3)  --> d1+d2+d3
        # data2: (d1, d2, d3)  --> d1+d2+d3
    distance = sq_distance ** 0.5
    sorted_distance_index_list = distance.argsort() # sort distance
    
    class_stats = {}
    for i in range(k):
        vote_label = labels[sorted_distance_index_list[i]]
        class_stats[vote_label] = class_stats.get(vote_label, 0) + 1
        
    sorted_class_stats = sorted(
        class_stats.items(), # an item: (label, value) --> index: 0 is label, 1 is value
        key=operator.itemgetter(1),
            # NOTE: The key parameter is always a function that is fed one item from the iterable
        reverse=True
    )
    
    label_output = sorted_class_stats[0][0] # [(label, value), (label, value), ...] 
        # choose the label with the max cnt value
    return label_output