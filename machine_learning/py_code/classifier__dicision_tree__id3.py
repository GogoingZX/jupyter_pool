"""
feature_name_list = ["is_big", "is_white"] --> is_big: (0, 1); is_white: (0, 1)
dataset = [
    [1, 1, "Y"],
    [1, 0, "N"],
    [1, 1, "Y"],
    [0, 1, "N"],
    [0, 1, "N"]
]
result_set = ["Y", "N"]
"""

import operator
from math import log

def calculate_shannon_entropy(dataset):
    entity_num = len(dataset) # data_set -- list_like
    label_stats = {}
    for entity_vector in dataset:
        current_label = entity_vector[-1] 
            # entity_vector -- list_like [d1, d2, ..., dn], assume db is the class
        if current_label not in label_stats.keys():
            label_stats[current_label] = 0
        else:
            pass
        label_stats[current_label] += 1

    shannon_entropy = 0
    for key in label_stats.keys(): # slightly slower than 'for key in label_stats:'
        label_prob = label_stats[key] / entity_num
        shannon_entropy = shannon_entropy - label_prob * log(label_prob, 2)
    
    return shannon_entropy


def split_dataset(
    dataset,
    axis,
    value
):
    new_dataset=[]
    for entity_vector in dataset:
        if entity_vector[axis] == value: # choose entity, entity_vector: [d1, d2, d3]
            reduced_entity_vector = entity_vector[:axis]
            reduced_entity_vector.extend(entity_vector[axis+1:]) # reduced_entity_vector: [d1, d3]
                # drop entity_vector[axis] and keep the original list unchanged
            new_dataset.append(reduced_entity_vector)
        else:
            pass
            
    return new_dataset


def choose_feature_to_split(dataset):
    feature_cnt = len(dataset[0]) - 1
    base_entropy = calculate_shannon_entropy(dataset)
    best_info_gain = 0
    best_feature_axis = -1 
        # create a default best feature
        # -1 means no best feature and choose the last one as the best feature
    
    for i in range(feature_cnt):
        feature_list = [entity_tmp[i] for entity_tmp in dataset]
        uniq_feature_set = set(feature_list)
        
        new_entropy = 0
        for feature_value in uniq_feature_set:
            sub_dataset = split_dataset(
                dataset=dataset,
                axis=i,
                value=feature_value
            )
            prob = len(sub_dataset) / len(dataset)
            new_entropy += prob * calculate_shannon_entropy(sub_dataset) # weighted
            
        info_gain = new_entropy - base_entropy
        if info_gain <= best_info_gain:
            best_info_gain = info_gain
            best_feature_axis = i
        else:
            pass
    
    return best_feature_axis   


def majority_vote(class_list):
    class_stats = {}
    for class_value in class_list:
        if class_value not in class_stats.keys():
            class_stats[class_value] = 0
        else:
            pass
        class_stats[class_value] += 1
    
    sorted_class_stats = sorted(
        class_stats.items(),
        key=operator.itemgetter(1),
        reverse=True
    )
    
    voted_class_value = sorted_class_stats[0][0]
    return voted_class_value


def create_decision_tree__id3(
    dataset,
    feature_name_list
):
    class_list = [entity_temp[-1] for entity_temp in dataset]
    
    if class_list.count(class_list[0]) == len(class_list): # leaf node -- stop condition 1
        return class_list[0]
    if len(dataset[0]) == 1: # traversed all features -- stop condition 2
        return majority_vote(class_list)
    
    feature_axis = choose_feature_to_split(dataset)
    feature_name = feature_name_list[feature_axis]
    tree = {
        feature_name: {}
    }
    del feature_name_list[feature_axis]
    
    feature_value_list = [entity_temp[feature_axis] for entity_temp in dataset]
    uniq_feature_set = set(feature_value_list)
    for feature_value in uniq_feature_set:
        sub_feature_name_list = feature_name_list[:]
        tree[feature_name][feature_value] = create_decision_tree__id3(
            dataset = split_dataset(
                dataset,
                axis=feature_axis,
                value=feature_value
            ),
            feature_name_list = sub_feature_name_list
        )
    
    return tree


"""
{
    "is_big": {
        0: "N",
        1: {
            "is_white": {
                0: "N",
                1: "Y"
            }
        }
    }
}

tree_dict = {"is_big": {0: "N", 1: {"is_white": {0: "N", 1: "Y"}}}}
"""

def get_leaf_qty(tree_dict):
    leaf_qty = 0
    first_key = list(tree_dict.keys())[0]
    sub_tree_dict = tree_dict[first_key]
    for key in sub_tree_dict.keys():
        if type(sub_tree_dict[key]).__name__ == "dict": # if type(sub_tree_dict[key]) is dict
            leaf_qty = leaf_qty + get_leaf_qty(sub_tree_dict[key])
        else:
            leaf_qty += 1
            
    return leaf_qty


def get_tree_depth(tree_dict):
    max_depth = 0
    first_key = list(tree_dict.keys())[0]
    sub_tree_dict = tree_dict[first_key]
    for key in sub_tree_dict.keys():
        if type(sub_tree_dict[key]) is dict:
            this_depth = 1 + get_tree_depth(sub_tree_dict[key])
        else:
            this_depth = 1
            
        if this_depth > max_depth:
            max_depth = this_depth
    
    return max_depth