import sys
import os
import time
import datetime
import pickle
import json
import re
import operator

from math import log
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("/Users/xuzhu/Desktop/code/assistants") # my package
from toolbox.os_assistant import scan_folder


DATA_FOLDER = "/Users/xuzhu/Desktop/data/open_dataset"
project_folder = os.path.join(DATA_FOLDER, "test__decision_tree")

raw_data = [
    [1, 1, "Y"],
    [1, 0, "N"],
    [1, 1, "Y"],
    [0, 1, "N"],
    [0, 1, "N"]
]
raw_df = pd.DataFrame(
    data=raw_data,
    columns=["is_big", "is_white", "good"]
)


def calculate_shannon_entropy(df):
    """
    Calculate Shannon entropy of the dataset
    """
    entity_qty = df.shape[0]
    class_stats = defaultdict(int)
    for index, row in df.iterrows():
        current_class = row[-1]
        class_stats[current_class] += 1
    
    shannon_entropy = 0
    for key in class_stats.keys():
        class_prob = class_stats[key] / entity_qty
        shannon_entropy = shannon_entropy - class_prob * log(class_prob, 2)
    
    return shannon_entropy


def split_dataset(
    df,
    feature_name,
    feature_value
):
    """
    Split dataset based on Shannon entropy
    
    If the entity's feature value = specific feature value
    ==> matched, grab this entity and delete this used feature to build a new dataset
    ==> all features only use once in this algorithm !!!
    """
    
    col_list = df.columns.to_list()
    col_list.remove(feature_name)
    new_df = pd.DataFrame(columns=col_list) # make sure the column position unchanged
    
    for index, row in df.iterrows(): # scan all rows
        if row[feature_name] == feature_value: 
            new_row = row.drop(feature_name)
            new_df = new_df.append(new_row, ignore_index=True) # Q: why does the col index change randomly?
        else: # not matched
            pass
    
    return new_df


def choose_best_feature_to_split(df):
    """
    Compare different features and choose the best one to split the dataset
    
    信息增益代表了在一个条件下, 信息复杂度(不确定性)减少的程度
    如果选择一个特征, 信息增益最大(信息不确定性减少的程度最大), 那么我们就选取这个特征
    信息增益 = 信息熵 - 条件熵
    ==> https://zhuanlan.zhihu.com/p/26596036
    """
    entropy__base = calculate_shannon_entropy(df)
    info_gain__max = 0
    
    # feature_qty = len(df.columns) - 1 # assume the last one column is the label
    feature_name_list = df.columns.to_list()[:-1] # exclude the label
    feature_qty = len(feature_name_list)
    best_feature_name = feature_name_list[-1]
    
    for feature_name in feature_name_list: # scan all feature columns
        # split dataset by using this feature, calculate the information gain
        current_feature_value_list = [row[feature_name] for index, row in df.iterrows()]
        uniq_feature_value_set = set(current_feature_value_list) # uniq value (or we can say 'class')
        
        entropy__condition = 0
        for feature_value in uniq_feature_value_set: # calculate the conditional entropy
            df__new = split_dataset(
                df=df,
                feature_name=feature_name,
                feature_value=feature_value
            )
            prob = df__new.shape[0] / df.shape[0] # calculate the probability of the subclass
            entropy__condition = entropy__condition + prob * calculate_shannon_entropy(df__new)
        
        info_gain = entropy__base - entropy__condition
        if info_gain > info_gain__max:
            info_gain__max = info_gain
            best_feature_name = feature_name
        else:
            pass
    
    return best_feature_name
            

def majority_vote(class_list):
    class_stats = {}
    for class_value in class_list:
        if class_value not in class_stats.keys():
            class_stats[class_value] = 0
        else:
            class_stats[class_value] += 1
        
    sorted_class_list = sorted(
        class_stats.items(),
        key=operator.itemgetter(1),
        reverse=True
    )
    # [(key, value), (key, value),... ]
    
    majority_class_value = sorted_class_list[0][0] # choose the key (class value) of the first element
    return majority_class_value


def create_decision_tree__id3(df):
    """
    Create a decision tree based on ID3 algorithm
    All features will be used once only
    """
    label_list = [row[-1] for index, row in df.iterrows()]
    if label_list.count(label_list[0]) == len(label_list): # stop condition 1: only 1 class ==> leaf node
        label = label_list[0]
        return label
    if len(df.iloc[0]) == 1: # stop condition 2: no other features
        # used all features but there still be several classes
        # need choose the majority class
        label = majority_vote(label_list)
        return label
    
    feature_name_list = df.columns.to_list()[:-1] # exclude the label
    feature_name = choose_best_feature_to_split(df)
    tree = {
        feature_name: {}
    }
    feature_name_list.remove(feature_name)
    
    feature_value_list = [row[feature_name] for index, row in df.iterrows()]
    uniq_feature_value_set = set(feature_value_list)
    for feature_value in uniq_feature_value_set:
        sub_df = split_dataset(
            df=df,
            feature_name=feature_name,
            feature_value=feature_value
        )
        
        tree[feature_name][feature_value] = create_decision_tree__id3(df=sub_df)
        
    return tree


def get_leaf_qty(tree_dict):
    """
    {'is_big': {0: 'N',
                1: {'is_white': {0: 'N',
                                 1: 'Y'}}}}
    """
    leaf_qty = 0
    first_feature = list(tree_dict.keys())[0]
    
    sub_tree_dict = tree_dict[first_feature]
    for key in sub_tree_dict.keys():
        if type(sub_tree_dict[key]).__name__ == "dict":
            leaf_qty += get_leaf_qty(sub_tree_dict[key])
        else: # stop condition ==> leaf node
            leaf_qty += 1
            
    return leaf_qty


def get_tree_depth(tree_dict):
    depth__max = 0
    first_feature = list(tree_dict.keys())[0]
    
    sub_tree_dict = tree_dict[first_feature]
    for key in sub_tree_dict.keys():
        if type(sub_tree_dict[key]) is dict:
            depth__current = 1 + get_tree_depth(sub_tree_dict[key])
        else:
            depth__current = 1
        
        if depth__current > depth__max:
            depth__max = depth__current
    
    return depth__max


def save_tree(
    input_tree,
    filepath
):
    with open(filepath, "wb") as f_write:
        pickle.dump(input_tree, f_write)


def load_tree(filepath):
    with open(filepath, "rb") as f_read:
        tree = pickle.load(f_read)
    
    return tree