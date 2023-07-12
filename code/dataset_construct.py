import os
import re
import pandas as pd
import pdb
import numpy as np

old_comp = ['graphics', 'operating system microsoft windows', 'ibm hardware', 'apple hardware', 'windows x', 'automobiles']
new_comp = ['graphics', 'windows', 'ibm', 'mac', 'x window', 'autos']

coarse_set = ['religion', 'computer', 'recreation', 'science', 'politics']
fine_set = ['atheism', 'graphics', 'windows', 'ibm', 'mac', 'x window', 'autos',
 'motorcycles', 'baseball', 'hockey', 'encryption', 'electronics', 'medicine',
 'space', 'christian', 'guns', 'mideast']
project_dict = {'computer': ['graphics', 'windows', 'ibm', 'mac', 'x window'],
                'politics': ['guns', 'mideast'],
                'recreation': ['autos', 'motorcycles', 'baseball', 'hockey'],
                'religion': ['atheism', 'christian'],
                'science': ['encryption', 'electronics', 'medicine', 'space']}
                
nyt_coarse_set = ["politics", "arts", "business", "science", "sports"]
nyt_fine_set = ["federal budget", "surveillance", "the affordable care act", "immigration", "law enforcement", "gay rights", "gun control", "military", "abortion", "dance", "television", "music", "movies", "stocks and bonds", "energy companies", "economy", "international business", "cosmos", "environment", "hockey", "basketball", "tennis", "golf", "football", "baseball", "soccer"]
nyt_proj_dict = {"politics": ["federal budget", "surveillance", "the affordable care act", "immigration", "law enforcement", "gay rights", "gun control", "military", "abortion"], "arts": ["dance", "television", "music", "movies"], "business": ["stocks and bonds", "energy companies", "economy", "international business"], "science": ["cosmos", "environment"], "sports": ["hockey", "basketball", "tennis", "golf", "football", "baseball", "soccer"]}

topic_template_dict = {1 : 'The topic of this post is ', 2 : 'They are discussing ', 3 : 'This post mainly talks about '}
nyt_topic_template_dict = {1 : 'The news is about ', 2 : 'The news is related to ', 3 : 'The topic of this passage is '}

weaksup_num = {'religion':75, 'computer':20, 'recreation':231, 'science': 25, 'politics':28}
nyt_weaksup_num = {"politics": 24, "arts": 46, "business": 33, "science": 21, "sports": 270}

weaksup_num_list = [75, 20, 231, 25, 28]
nyt_weaksup_num_list = [24, 46, 33, 21, 270]

fine_dict = dict(zip(fine_set, [i+len(coarse_set) for i in range(len(fine_set))]))
coarse_dict = dict(zip(coarse_set, [i for i in range(len(coarse_set))]))

nyt_gloss_path = '../data/nyt_fine_definition.csv'
nyt_gloss_coarse_path = '../data/nyt_coarse_definition.csv'

nyt_gloss = pd.read_csv(nyt_gloss_path)
nyt_coarse_gloss = pd.read_csv(nyt_gloss_coarse_path)
print(nyt_coarse_gloss)

gloss_path = '../data/20news_fine_definition.csv'
gloss_coarse_path = '../data/20news_coarse_definition.csv'

gloss = pd.read_csv(gloss_path)
coarse_gloss = pd.read_csv(gloss_coarse_path)

# try different surface names
def convert_label_name(df):
    new_df = df.copy()
    new_names = []
    for i in range(len(new_df)):
        if new_df['fine_label'][i] in old_comp:
            index = old_comp.index(new_df['fine_label'][i])
            new_names.append(new_comp[index])
        else:
            new_names.append(new_df['fine_label'][i])
    new_df['fine_label'] = new_names
    print(new_df)
    return new_df
    
def from_label_name_to_definition(data_name, label_name):
    if 'nyt' in data_name:
        label_index = nyt_gloss['fine_label'].tolist().index(label_name)
        return nyt_gloss['fine_definition'][label_index]
    else:
        label_index = gloss['fine_label'].tolist().index(label_name)
        return gloss['fine_definition'][label_index]

def from_coarse_name_to_definition(data_name, label_name):
    if 'nyt' in data_name:
        label_index = nyt_coarse_gloss['coarse_label'].tolist().index(label_name)
        return nyt_coarse_gloss['coarse_definition'][label_index]
    else:
        label_index = coarse_gloss['coarse_label'].tolist().index(label_name)
        return coarse_gloss['coarse_definition'][label_index]

def meaning_construction(data_name, template_index=1):
    coarse_meanings = []
    meanings = []
    if 'nyt' in data_name:
        for coarse_label in nyt_coarse_set:
            def_str = from_coarse_name_to_definition(data_name, coarse_label)
            label_str = nyt_topic_template_dict[template_index] + coarse_label + '. '
            # def_str = coarse_label + ' means ' + def_str
            total_str = label_str + def_str
            coarse_meanings.append(total_str)
        for fine_label in nyt_fine_set:
            def_str = from_label_name_to_definition(data_name, fine_label)
            label_str = nyt_topic_template_dict[template_index] + fine_label + '. '
            # def_str = fine_label + ' means ' + def_str
            total_str = label_str + def_str
            meanings.append(total_str)
        return coarse_meanings, meanings
    else:
        for coarse_label in coarse_set:
            def_str = from_coarse_name_to_definition(data_name, coarse_label)
            label_str = topic_template_dict[template_index] + coarse_label + '. '
            # def_str = coarse_label + ' means ' + def_str
            total_str = label_str + def_str
            coarse_meanings.append(total_str)
        for fine_label in fine_set:
            def_str = from_label_name_to_definition(data_name, fine_label)
            label_str = topic_template_dict[template_index] + fine_label + '. '
            # def_str = fine_label + ' means ' + def_str
            total_str = label_str + def_str
            meanings.append(total_str)
    return coarse_meanings, meanings

coarse_meaning, fine_meaning = meaning_construction('20news', template_index=1)
nyt_coarse_meaning, nyt_fine_meaning = meaning_construction('nyt', template_index=3)

meaning_df = pd.DataFrame()
meaning_df['meaning'] = coarse_meaning 
meaning_df['label'] = coarse_set
meaning_df.to_csv('../data/20news_coarse_gloss_transfer.csv')

meaning_df = pd.DataFrame()
meaning_df['meaning'] = fine_meaning
meaning_df['label'] = fine_set
meaning_df.to_csv('../data/20news_fine_gloss_transfer.csv')

meaning_df = pd.DataFrame()
meaning_df['meaning'] = nyt_coarse_meaning
meaning_df['label'] = nyt_coarse_set
meaning_df.to_csv('../data/nyt_coarse_gloss_transfer.csv')

meaning_df = pd.DataFrame()
meaning_df['meaning'] = nyt_fine_meaning
meaning_df['label'] = nyt_fine_set
meaning_df.to_csv('../data/nyt_fine_gloss_transfer.csv')




nyt_coarse_gloss = [coarse_label + ' ' + from_coarse_name_to_definition('nyt', coarse_label) for coarse_label in nyt_coarse_set]
coarse_gloss = [coarse_label + ' ' + from_coarse_name_to_definition('20news', coarse_label) for coarse_label in coarse_set] 

nyt_fine_gloss = [fine_label + ' ' + from_label_name_to_definition('nyt', fine_label) for fine_label in nyt_fine_set]
fine_gloss = [fine_label + ' ' + from_label_name_to_definition('20news', fine_label) for fine_label in fine_set]


def split_train_valid_test(df):
    df_new = df.sample(frac=1).dropna()
    test_size = int(len(df_new) / 10)
    valid_size = test_size
    train_size = len(df_new) - test_size * 2
    valid_size = train_size + test_size
    return df_new[:train_size], df_new[train_size:valid_size], df_new[valid_size:]
    
def convert_to_numbers(df):
    clabels = df['coarse_label'].tolist()
    flabels = df['fine_label'].tolist()
    clabels = [coarse_set.index(coarse) for coarse in clabels]
    flabels = [fine_set.index(fine) for fine in flabels]
    df['coarse_label'] = clabels
    df['fine_label'] = flabels
    return df
    
def construct_numerate_dict():
    numerate_dict = {}
    for coarse in project_dict:
        numerate_dict[coarse_set.index(coarse)] = [fine_set.index(fine) for fine in project_dict[coarse]]
    print(numerate_dict)
    return numerate_dict

def construct_numerate_dict_nyt():
    numerate_dict = {}
    for coarse in nyt_proj_dict:
        numerate_dict[nyt_coarse_set.index(coarse)] = [nyt_fine_set.index(fine) for fine in nyt_proj_dict[coarse]]
    print(numerate_dict)
    return numerate_dict

print(construct_numerate_dict())
print(construct_numerate_dict_nyt())

def convert_to_multilabel(df):
    text = df['text']
    clabels = df['coarse_label']
    flabels = df['fine_label']
    multi_labels = []
    label_names = []
    for clabel in clabels:
        multi_label = [float(i == coarse_dict[clabel]) for i in range(len(coarse_set))]
        multi_label.extend([float(i in project_dict[clabel]) for i in fine_set])
        label_name = [clabel] + project_dict[clabel]
        multi_labels.append(multi_label)
        label_names.append(label_name)
    df_multi = pd.DataFrame()
    df_multi['text'] = text
    df_multi['multi_label'] = multi_labels
    df_multi['fine_label'] = flabels
    return df_multi
    
def select_initial_seed(coarse_df, seed_df):
    seed_index = []
    for text in seed_df.text:
        try:
            index = coarse_df.text.tolist().index(text)
            seed_index.append(index)
        except:
            pass
    return seed_index