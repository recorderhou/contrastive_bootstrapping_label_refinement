# with feature
import os 
import torch
import math
import pdb
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
import json
from collections import Counter

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torch.utils.data as Data
import torch.nn as nn
from torch.optim import *
import copy
import re
import os
import pandas as pd
import nltk
import glob
import numpy as np
from transformers import DataCollatorWithPadding
from torch.nn.functional import cosine_similarity, one_hot
from torch.nn import TripletMarginWithDistanceLoss
from dataset_construct import *
from sklearn.metrics import accuracy_score, f1_score
import random
import faiss
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from faiss import normalize_L2


import argparse

parser = argparse.ArgumentParser(description='error analysis argument')
parser.add_argument('--dataset', default='20news.csv', type=str, help='dataset, nyt or 20news')
parser.add_argument('--gama', default=0.05, type=float, help='triplet loss of pos_neg')
parser.add_argument('--lambdas', default=0.05, type=float, help='triplet loss of select')
parser.add_argument('--beta', default=0.05, type=float, help='threshold of select')
parser.add_argument('--maxbeta', default=0.9, type=float, help='threshold of select in max mode')
parser.add_argument('--alphac', default=0.05, type=float, help='loss of coarse')
parser.add_argument('--alphaf', default=1.0, type=float, help='loss of coarse')
parser.add_argument('--alphas', default=0.1, type=float, help='loss of coarse2')
parser.add_argument('--use_pos', action='store_true', help='basic triplet loss')
parser.add_argument('--use_select', action='store_true', help='use the outstanding fine label as next iter\'s ground truth')
parser.add_argument('--use_adapter', action='store_true', help='whether to change the threshold or not')
parser.add_argument('--adapter', default=0.05, type=float, help='change the threshold base on former similarity, only select the top')
parser.add_argument('--use_coarse', action='store_true', help='use the relationship between coarse and fine labels to improve the representation')
parser.add_argument('--coarse_type', default=1, type=int, help='choose between 2 ways of using coarse label')
parser.add_argument('--use_labeled', action='store_true', help='use some selected ground truth as the initial seed')
parser.add_argument('--labeled_portion', default='005', type=str, help='choose the portion of labeled data')
parser.add_argument('--data_dir', default='../data', type=str, help='dataset_dir')
parser.add_argument('--times', default='1', type=str, help='dataset_dir')
parser.add_argument('--coarse_label', default='recreation', choices=['religion', 'computer', 'recreation', 'science', 'politics'])
parser.add_argument('--use_coarse_label', action='store_true', help='whether to develop a seperate classifier for a certain coarse label or not')
parser.add_argument('--epoch_num', default=5, type=int, help='total epoch num that the model trained on')
parser.add_argument('--select_epoch', default=1, type=int)
parser.add_argument('--use_csls', action='store_true', help='whether to use csls to replace cos')
parser.add_argument('--use_infonce', action='store_true', help='whether to use infonce to replace margin loss')
parser.add_argument('--T', default=1.0, help='softmax temperature')
parser.add_argument('--use_label_repr', action='store_true', help='initialize the label representation using PLM, then use it as a parameter to update')
parser.add_argument('--select_mode', default='margin', choices=['margin', 'max'], help='select mode, margin is to larger max - second max, max is to larger max only')
parser.add_argument('--prediction_mode', default='cos', choices=['cos', 'csls'], help='prediction mode, use cosine similarity or csls distance')
parser.add_argument('--use_hard', action='store_true', help='wether to select the hardest negative sample within a batch')
parser.add_argument('--use_weaksup', action='store_true', help='wether to use the initial weak supervision')
parser.add_argument('--use_gloss', action='store_true', help='wether to add definition to the label surface name')
parser.add_argument('--seed_val', default=42, type=int)
parser.add_argument('--soft_weaksup', action='store_true', help='how to deal with weaksup-lglobal conflict')
parser.add_argument('--weaksup_select', default='full', choices=['full', 'coarse'], help='how to select')
args = parser.parse_args()

if type(args.gama) == type([3.0]):
    args.gama = args.gama[0]
if type(args.adapter) == type([3.0]):
    args.adapter = args.adapter

print(args)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
numerate_dict = construct_numerate_dict()
best_acc = 0.0
best_test_acc = 0.0
train_num = 0
valid_num = 0
test_num = 0
seed_index = []
ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCEWithLogitsLoss()
pos_loss = nn.MarginRankingLoss(margin=args.gama)
coarse_loss = nn.MarginRankingLoss(margin=args.alphac)
dataset_name = '20news'
weaksup_num = 0
weaksup_fine = []
weaksup_text = []
weaksup_coarse = []
    
# writer = SummaryWriter('./runs')

if 'nyt' in args.dataset:
    coarse_set = nyt_coarse_set
    fine_set = nyt_fine_set
    numerate_dict = construct_numerate_dict_nyt()
    dataset_name = 'nyt'
    fine_gloss = nyt_fine_gloss
    coarse_gloss = nyt_coarse_gloss
    fine_meaning = nyt_fine_meaning
    coarse_meaning = nyt_coarse_meaning
    
coarse_size = len(coarse_set)
fine_size = len(fine_set)

coarse_size = len(coarse_set)


def create_ckpt_name(args):
    if 'nyt' in args.dataset:
        init = 'nyt'
    else:
        init = '20news'
    if args.use_infonce:
        init = init + '_' + 'nce' + '_' + str(args.T)
    if args.use_csls:
        init = init + '_' + 'csls'
    if args.use_pos:
        init = init + '_' + 'pos'
        init = init + '_' + str(args.gama)
    if args.use_select:
        init = init + '_' + 'select'
        init = init + '_' + str(args.lambdas) + '_' + str(args.select_epoch)
        if args.use_adapter:
            init = init + '_' + 'a' + str(args.adapter)
        else:
            init = init + '_' + str(args.beta)
        if args.select_mode == 'margin':
            init = init + '_margin'
        else:
            init = init + '_max'
    if args.use_coarse:
        init = init + '_' + 'coarse' + str(args.coarse_type)
        if args.coarse_type == 1:
            init = init + '_' + str(args.alphac)
        else:
            init  = init + '_' + str(args.alphas)
        # init = init + '_' + str(args.alphac) + '_' + str(args.alphaf)
    if args.use_weaksup:
        init = init + '_' + 'weaksup'
        # init = init + '_' + str(args.use_labeled)
        # init = init + '_' + '005'
    if args.use_gloss:
        init = init + '_' + 'gloss'
    if args.use_coarse_label:
        init = init + '_' + 'onlycoarse'
        init = init + '_' + str(args.coarse_label)
    init = init + '_' + args.times
    return init

    
ckpt_name = create_ckpt_name(args)
print('ckpt_name is', ckpt_name)

# code from 
# https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
# compute cosine similarity
# checked
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def euclidean_distance(x, y):
    """
    Computes the Manhattan distance between two arrays using PyTorch.
    """
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)

    # Compute the absolute difference between x and y
    diff = torch.abs(x - y) ** 2

    # Sum over the last dimension to get the Manhattan distance
    dist = torch.sum(diff, dim=-1)
    
    sim = 1 / (1 + dist)

    return sim
    
def elucidian_sim_matrix(a, b, eps=1e-8):
    return euclidean_distance(a, b)

# get arg2's knn in arg1(use arg2 as query, select k instances from arg1)
# checked
def get_nn_avg_dist(fine_hidden, text_hidden, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    fine_hidden = fine_hidden.clone().detach().cpu().numpy()
    text_hidden = text_hidden.clone().detach().cpu().numpy()
    normalize_L2(fine_hidden)
    normalize_L2(text_hidden)
    index = faiss.IndexFlatIP(fine_hidden.shape[1])
    index.add(fine_hidden)
    distances, ids = index.search(text_hidden, knn)
    # print(distances)
    return ids
    
def euclidean_get_nn_avg_dist(fine_hidden, text_hidden, knn):
    # pdb.set_trace()
    fine_hidden = fine_hidden.clone().detach()
    text_hidden = text_hidden.clone().detach()
    # Compute pairwise distances between query and database vectors
    distances = euclidean_distance(text_hidden, fine_hidden)

    # Find the k nearest neighbors for each query vector
    _, indices = torch.topk(distances, k=knn, dim=1, largest=True)

    return indices

def manhattan_get_nn_avg_dist(fine_hidden, text_hidden, knn):
    fine_hidden = fine_hidden.clone().detach().cpu().numpy()
    text_hidden = text_hidden.clone().detach().cpu().numpy()
    # normalize_L2(fine_hidden)
    # normalize_L2(text_hidden)
    index = faiss.IndexFlatL1(fine_hidden.shape[1])
    index.add(fine_hidden)
    distances, ids = index.search(text_hidden, knn)
    # print(distances)
    return ids

# CSLS
# for each batch of instances(text), we want to compute their logits(batch_size * fine_label_size)
# for each instance, its logits is: cos(text, label) - mean(KNN(text)) for label in fine_label_set
# the fine label set include all fine labels
def NCE_CSLS(fine_hidden, text_hidden):
    # pdb.set_trace()
    knn = 5
    true_fine = []
    false_fine = []
    false_fines = []
    # batch_size, label_size
    origin_distance = sim_matrix(text_hidden, fine_hidden)
    label_num = fine_hidden.shape[0]
    text_num = text_hidden.shape[0]
    
    
    # batch_size, knn
    avg_text_knn_id = torch.from_numpy(get_nn_avg_dist(fine_hidden, text_hidden, knn))
    # batch_size * knn, 768
    top_label_hidden = torch.index_select(fine_hidden, 0, avg_text_knn_id.reshape(-1).cuda())
    # batch_size * batch_size, knn
    top_label_distance = sim_matrix(text_hidden, top_label_hidden).reshape(-1, knn)
    # batch_size * knn -> batch_size
    top_label_distance = torch.index_select(top_label_distance, 0, torch.arange(0, top_label_distance.shape[0], text_num+1).cuda()).mean(-1)
    origin_distance.sub_(top_label_distance.unsqueeze(1))
    
    return origin_distance   

def manhattan_CSLS(fine_hidden, text_hidden):
    # pdb.set_trace()
    knn = 5
    true_fine = []
    false_fine = []
    false_fines = []
    # batch_size, label_size
    origin_distance = manhattan_sim_matrix(text_hidden, fine_hidden)
    label_num = fine_hidden.shape[0]
    text_num = text_hidden.shape[0]
    
    
    # batch_size, knn
    avg_text_knn_id = torch.from_numpy(manhattan_get_nn_avg_dist(fine_hidden, text_hidden, knn))
    # batch_size * knn, 768
    top_label_hidden = torch.index_select(fine_hidden, 0, avg_text_knn_id.reshape(-1).cuda())
    # batch_size * batch_size, knn
    top_label_distance = manhattan_sim_matrix(text_hidden, top_label_hidden).reshape(-1, knn)
    # batch_size * knn -> batch_size
    top_label_distance = torch.index_select(top_label_distance, 0, torch.arange(0, top_label_distance.shape[0], text_num+1).cuda()).mean(-1)
    origin_distance.sub_(top_label_distance.unsqueeze(1))
    
    return origin_distance   
    
def euclidean_CSLS(fine_hidden, text_hidden):
    # pdb.set_trace()
    knn = 5
    true_fine = []
    false_fine = []
    false_fines = []
    # batch_size, label_size
    origin_distance = euclidean_sim_matrix(text_hidden, fine_hidden)
    label_num = fine_hidden.shape[0]
    text_num = text_hidden.shape[0]
    
    
    # batch_size, knn
    avg_text_knn_id = euclidean_get_nn_avg_dist(fine_hidden, text_hidden, knn)
    # batch_size * knn, 768
    top_label_hidden = torch.index_select(fine_hidden, 0, avg_text_knn_id.reshape(-1).cuda())
    # batch_size * batch_size, knn
    top_label_distance = euclidean_sim_matrix(text_hidden, top_label_hidden).reshape(-1, knn)
    # batch_size * knn -> batch_size
    top_label_distance = torch.index_select(top_label_distance, 0, torch.arange(0, top_label_distance.shape[0], text_num+1).cuda()).mean(-1)
    origin_distance.sub_(top_label_distance.unsqueeze(1))
    
    return origin_distance  
        

# just a simple infonce
class RobertaModelForLabelNCE(nn.Module):
    def __init__(self, T=1.0):
        super(RobertaModelForLabelNCE, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base", 
                                                    output_hidden_states=True)
        self.fine_label_linear = nn.Linear(17, 768, bias=False)
        self.coarse_label_linear = nn.Linear(5, 768, bias=False)
        self.initialize_label_representation()
        self.T = T
    
    def initialize_label_representation(self):
        # pdb.set_trace()
        tokenized_coarse = tokenizer(coarse_set, padding='longest')
        tokenized_labels = tokenizer(fine_set, padding='longest')
        fine_initial = self.roberta(torch.tensor(tokenized_labels['input_ids']), torch.tensor(tokenized_labels['attention_mask'])).last_hidden_state[:,0,:].detach().requires_grad_(True)
        coarse_initial = self.roberta(torch.tensor(tokenized_coarse['input_ids']), torch.tensor(tokenized_coarse['attention_mask'])).last_hidden_state[:,0,:].detach().requires_grad_(True)
        self.fine_label_linear.weight = nn.Parameter(fine_initial)
        self.coarse_label_linear.weight = nn.Parameter(coarse_initial)

    def forward(self, batched_data):
        # text, fine labels, coarse labels
        (input_ids, attention_mask), (fine_ids, fine_attention), (coarse_ids, coarse_attention), true_coarses = batched_data
        inputs = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        
        # since every instance has a set of coarse labels and fine labels, we only need to pick one
        fine_ids = fine_ids[0].cuda()
        fine_attention = fine_attention[0].cuda()
        
        coarse_ids = coarse_ids[0].cuda()
        coarse_attention = coarse_attention[0].cuda()

        candidate_fine_set = [numerate_dict[true_coarse.item()] for true_coarse in true_coarses]

        # print(candidate_fine_set)
        
        roberta_outputs = self.roberta(inputs, attention_mask)
        fine_outputs = self.roberta(fine_ids, fine_attention)
        coarse_outputs = self.roberta(coarse_ids, coarse_attention)

        # pdb.set_trace()
        
        # (batch_size, sequence_length, hidden_size)
        # (batch_size, hidden_size)
        sequence_output = roberta_outputs.last_hidden_state[:,0,:]
        fine_output = fine_outputs.last_hidden_state[:, 0, :]
        coarse_output = coarse_outputs.last_hidden_state[:, 0, :]
        
        batch_size = sequence_output.shape[0]

        logits_fine = torch.einsum('nc,kc->nk', [sequence_output, fine_output])
        logits_coarse = torch.einsum('nc,kc->nk', [sequence_output, coarse_output])
        target_fine = torch.zeros_like(logits_fine)
        target_coarse = true_coarses
        
        # pdb.set_trace()

        max_index_num = max([len(can_fines) for can_fines in candidate_fine_set])
        index = [can_fines + [can_fines[-1]] * (max_index_num - len(can_fines)) for can_fines in candidate_fine_set]
        
        # pdb.set_trace()
        
        # pdb.set_trace()
        if args.use_csls:
            logits_fine = NCE_CSLS(fine_output, sequence_output)
            # pdb.set_trace()
            logits_coarse = NCE_CSLS(coarse_output, sequence_output)
        
        target_fine.scatter_(1, torch.tensor(index).cuda(), 1)
        target_fine = target_fine.cuda()
        target_coarse = target_coarse.cuda()

        # pdb.set_trace()
        if args.use_infonce:
        # apply temperature
            logits_fine /= self.T
            logits_coarse /= self.T

        return sequence_output, fine_output, coarse_output, logits_fine, target_fine, logits_coarse, target_coarse

# loss function     
def criterion(text_hidden, fine_hidden, coarse_hidden, true_coarse, ground_fine, logits, target, indexs, coarse_logits, coarse_target, epoch=0, mode='train'):
    global weaksup_num
    batch_size = text_hidden.shape[0]
    true_coarse = true_coarse.numpy()
    true_fine = [numerate_dict[t_coarse] for t_coarse in true_coarse]
    loss = torch.tensor(0.0)
    pos_neg_loss = torch.tensor(0.0)
    coarse_fine_loss = torch.tensor(0.0)
    select_loss = torch.tensor(0.0)
    labeled_loss = torch.tensor(0.0)
    seed_loss = torch.tensor(0.0)
    # pos, coarse, select, seed
    loss_size = [0.0, 0.0, 0.0, 0.0]
    return_static = []
    selected_mask_list = []
    batch_size = text_hidden.shape[0]

    if args.use_pos:
        if args.use_infonce:
            pos_neg_loss = bce_loss(logits, target)
    if args.use_coarse:
        if args.use_infonce:
            coarse_fine_loss = ce_loss(coarse_logits, coarse_target)

    for i in range(batch_size):
        false_fine = [j for j in range(fine_size) if j not in true_fine[i]]
        selected_false = np.random.choice(false_fine, size=len(true_fine[i]))
        false_labels = torch.index_select(fine_hidden, 0, torch.tensor(selected_false).cuda())
        true_labels = torch.index_select(fine_hidden, 0, torch.tensor(true_fine[i]).cuda())
        if args.use_csls:
            positive_dist = torch.index_select(logits[i], 0, torch.tensor(true_fine[i]).cuda())
            if args.use_hard:
                selected_false_logits = torch.index_select(logits[i], 0, torch.tensor(false_fine).cuda())
                negative_dist = torch.topk(selected_false_logits, len(true_fine[i]), largest=True).values
            else:
                negative_dist = torch.index_select(logits[i], 0, torch.tensor(selected_false).cuda())
        else:
            # positive_dist = cosine_similarity(text_hidden[i].unsqueeze(0), true_labels)
            positive_dist = torch.cdist(text_hidden[i], true_labels, p=2)
            if args.use_hard:
                total_negative_hidden = torch.index_select(fine_hidden, 0, torch.tensor(false_fine).cuda())
                # total_negative_sim = cosine_similarity(text_hidden[i].unsqueeze(0), total_negative_hidden)
                total_negative_sim = torch.cdist(text_hidden[i], total_negative_hidden, p=2)
                negative_dist = torch.topk(total_negative_sim, len(true_fine[i]), largest=True).values
            else:
                # negative_dist = cosine_similarity(text_hidden[i].unsqueeze(0), false_labels)
                negative_dist = torch.cdist(text_hidden[i], false_labels, p=2)
        return_static.append([positive_dist.clone().cpu().detach().numpy().tolist(), negative_dist.clone().cpu().detach().numpy().tolist(), true_fine[i], selected_false.tolist(), true_coarse[i]])
        
        # use margin loss
        # we want sampled negative instance have at least 
        if args.use_pos and not args.use_infonce:
            output = pos_loss(positive_dist, negative_dist, torch.tensor([1 for _ in range(len(true_fine[i]))]).cuda())
            pos_neg_loss = pos_neg_loss + output
        
        if args.use_coarse and not args.use_infonce:
            # pdb.set_trace()
            false_coarses = [j for j in range(coarse_size) if j != true_coarse[i]]
            selected_false_coarse = np.random.choice(false_coarses, size=1)
            false_coarse_label = torch.index_select(coarse_hidden, 0, torch.tensor(selected_false_coarse).cuda())
            # candidate fine labels should not be close to their coarse label
            true_coarse_label = torch.index_select(coarse_hidden, 0, torch.tensor(true_coarse[i]).cuda())
            true_labels = torch.index_select(fine_hidden, 0, torch.tensor(true_fine[i]).cuda())
            if args.use_csls:
                # pdb.set_trace()
                if args.use_hard:
                    total_negative_coarses = torch.index_select(coarse_logits[i], 0, torch.tensor(false_coarses).cuda())
                    negative_coarse = torch.topk(total_negative_coarses, 1, largest=True).values
                else:
                    negative_coarse = torch.index_select(coarse_logits[i], 0, torch.tensor(selected_false_coarse).cuda())
                positive_fine = torch.index_select(logits[i], 0, torch.tensor(true_fine[i]).cuda())
                positive_coarse = torch.index_select(coarse_logits[i], 0, torch.tensor(true_coarse[i]).cuda())
            else:
                if args.use_hard:
                    total_negative_hidden = torch.index_select(coarse_hidden, 0, torch.tensor(false_coarses).cuda())
                    # total_negative_coarses = cosine_similarity(text_hidden[i].unsqueeze(0), total_negative_hidden)
                    total_negative_coarses = torch.cdist(test_hidden[i], total_negative_hidden, p=2)
                    negative_coarse = torch.topk(total_negative_coarses, 1, largest=True).values
                else:
                    # negative_coarse = cosine_similarity(text_hidden[i].unsqueeze(0), false_coarse_label)
                    negative_coarse = torch.cdist(text_hidden[i], false_coarse_label, p=2)
                # positive_coarse = cosine_similarity(text_hidden[i].unsqueeze(0), true_coarse_label)
                positive_coarse = torch.cdist(text_hidden[i], true_coarse_label, p=2)
                # positive_fine = cosine_similarity(text_hidden[i].unsqueeze(0), true_labels)
                positive_fine = torch.cdist(text_hidden[i], true_labels, p=2)
            negative_mean = torch.mean(negative_coarse)
            negative_abs = negative_coarse @ negative_coarse
            # pdb.set_trace()
            if args.coarse_type == 1:
                # coarse_output = torch.clamp(negative_coarse - positive_coarse + args.alphac, min=0.0).sum()
                coarse_output = coarse_loss(positive_coarse, negative_coarse, torch.tensor([1]).cuda())
            else:
                coarse_output = torch.clamp(torch.abs(negative_coarse - negative_mean), min=0.0).sum() + args.alphas * negative_abs
            coarse_fine_loss = coarse_fine_loss + coarse_output
        
        # max - second max after training args.select_epoch epochs
        # if select mode == max, then we want the max prediction as big as possible
        # if select mode == margin, then we want max - second max as big as possible
        # mix this two?
        # what about larger the max's margin and the max margin's max?
        if args.use_select and epoch >= args.select_epoch:
            select_output = 0.0
            sorted_pos = torch.sort(positive_dist, descending=True)
            if args.select_mode == 'margin':
                if sorted_pos.values[0] - sorted_pos.values[1] > args.beta:
                    # pdb.set_trace()
                    select_output = torch.clamp(sorted_pos.values[1] - sorted_pos.values[0] + args.lambdas, min=0.0)
            else:
                if sorted_pos.values[0] > args.maxbeta:
                    select_output = 1.0 - sorted_pos.values[0]
                pdb.set_trace()
            if select_output == 0.0:
                pass
            elif select_output.item() < 0:
                pdb.set_trace()
            select_loss = select_loss + select_output
        
        # if args.use_weaksup and epoch == 0:
        # if args.use_weaksup:
        if args.use_weaksup:
            if args.use_select and epoch >= args.select_epoch:
                continue
            seed_output = 0.0
            if int(indexs[i]) in seed_index:
                # pdb.set_trace()
                index_in_seed = seed_index.index(int(indexs[i]))
                weaksup_ground = fine_set.index(weaksup_fine[index_in_seed])
                weaksup_num += 1
                sorted_pos = torch.sort(positive_dist, descending=True)
                ground_dist = torch.index_select(positive_dist, 0, torch.tensor(true_fine[i].index(weaksup_ground)).cuda())
                if sorted_pos.indices[0] == true_fine[i].index(weaksup_ground):
                    seed_output = torch.clamp(sorted_pos.values[1] - sorted_pos.values[0] + args.lambdas, min=0.0)
                else:
                    if args.soft_weaksup:
                        pass
                    else:
                        seed_output = torch.clamp(sorted_pos.values[0] - ground_dist + args.lambdas, min=0.0)
            seed_loss = seed_loss + seed_output
        
    loss = coarse_fine_loss + pos_neg_loss + select_loss + seed_loss

    loss_size = [coarse_fine_loss, pos_neg_loss, select_loss, seed_loss]
        
    return loss, return_static
  
# choose the most similar one as predict label
def prediction(text_hidden, label_hidden, true_coarse):
    batch_size = text_hidden.shape[0]
    true_coarse = true_coarse.numpy()
    true_fine = [numerate_dict[t_coarse] for t_coarse in true_coarse]
    preds = [0 for _ in range(batch_size)]
    
    if args.prediction_mode == 'csls':
        total_sims = euclidean_CSLS(label_hidden, text_hidden)
    
    for i in range(batch_size):
        true_labels = torch.index_select(label_hidden, 0, torch.tensor(true_fine[i]).cuda())
        if args.prediction_mode == 'cos':
            # sims = cosine_similarity(text_hidden[i].unsqueeze(0), true_labels)
            # sims = torch.cdist(text_hidden[i], true_labels, p=2)
            # pdb.set_trace()
            sims = euclidean_distance(text_hidden[i].unsqueeze(0), true_labels)
        else:
            sims = torch.index_select(total_sims[i], 0, torch.tensor(true_fine[i]).cuda())
        preds[i] = true_fine[i][torch.argmax(sims, -1).item()]
    return preds

def train(train_dataloader, model, optimizer, scheduler, epoch):
    train_loss_total = 0.0
    labeled_train_iter = iter(train_dataloader)
    analyze_result = []
    # store all the best - second best
    epoch_better = []
    
    model.train()
    total_steps = 0
    epoch_loss = 0.0
    for (batch_idx, batched_input) in enumerate(train_dataloader):

        total_steps += 1
        
        # text_tensor, fine_tensor, coarse_tensor, true_labels, true_fines, pred_ground, idx, text
        batch_size = batched_input[0][0].shape[0]
        text_tensor = batched_input[0]
        fine_labels = batched_input[1]
        coarse_labels = batched_input[2]
        # the coarse labels that we need to know
        labels = batched_input[3]
        # true fine label
        true_label = batched_input[4]
        indexs = batched_input[5]
        texts = batched_input[6]
        selected_mask_list = []
        # pdb.set_trace()
        
        # sequence_output, fine_output, coarse_output, logits_fine, target_fine, logits_coarse, target_coarse
        text_hidden, fine_hidden, coarse_hidden, logits, target, c_logits, c_target = model(batched_input[:4])
        
        # anchor, positive, negative
        
        dataset_index = batched_input[-2]
        
        # add index, already changed
        # add true fine
        loss, dist = criterion(text_hidden, fine_hidden, coarse_hidden, labels, true_label, logits, target, indexs, c_logits, c_target, epoch)
        
        # pdb.set_trace()
        train_loss_total += loss.item()
        
        # ground_labels[dataset_index] = pred_label
        optimizer.zero_grad()
        # pdb.set_trace()
        if type(loss) == type(torch.tensor(0.0)):
            loss.backward()
        else:
            pdb.set_trace()
            
        optimizer.step()
        scheduler.step()
        # print(batch_idx)

        if batch_idx % 200 == 0:
            print("epoch {}, step {}, loss {}".format(
                epoch, batch_idx, loss.item()))
            # pdb.set_trace()
           
    train_loss_total /= len(train_dataloader)
    print("epoch {}, avg. loss {}".format(epoch, train_loss_total))

    return analyze_result
                
                
class PlainData(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_fine, dataset_coarse, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.fines = dataset_fine
        self.coarses = dataset_coarse
        self.max_seq_len = max_seq_len
        # self.pred_ground_truth = select_initial_seed(self.fines)

    def __len__(self):
        return len(self.fines)

    def __getitem__(self, idx):
        text = self.text[idx]
        fine = self.fines[idx]
        coarse = self.coarses[idx]
        # pred_ground = torch.tensor(self.pred_ground_truth[idx])
        tokenized_labels = []
        try:
            tokenized_text = self.tokenizer(text, truncation=True, padding='max_length', max_length=512)
        except:
            pdb.set_trace()
        if args.use_gloss:
            # tokenized_labels = self.tokenizer(fine_gloss, padding='longest')
            # tokenized_coarse = self.tokenizer(coarse_gloss, padding='longest')
            tokenized_labels = self.tokenizer(fine_meaning, padding='longest')
            tokenized_coarse = self.tokenizer(coarse_meaning, padding='longest')
        else:
            tokenized_labels = self.tokenizer(fine_set, padding='longest')
            tokenized_coarse = self.tokenizer(coarse_set, padding='longest')
        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']
        text_tensor = (torch.tensor(input_ids), torch.tensor(attention_mask))
        fine_tensor = (torch.tensor(tokenized_labels['input_ids']), torch.tensor(tokenized_labels['attention_mask']))
        coarse_tensor = (torch.tensor(tokenized_coarse['input_ids']), torch.tensor(tokenized_coarse['attention_mask']))
        true_coarses = torch.tensor(coarse_set.index(coarse))
        true_fines = torch.tensor(fine_set.index(fine))
        # tokenized text, tokenized fine label set, tokenized coarse label set, true coarse label, true fine label, existing "true" labels
        return text_tensor, fine_tensor, coarse_tensor, true_coarses, true_fines, idx, text

# test
def validate(valloader, model, epoch, mode='test'):
    global seed_index
    analyze_result = []
    # store all the best - second best
    epoch_better = []
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        macros = 0
        micros = 0
        
        tmp_df = pd.DataFrame()
        text = []
        c_label = []
        f_label = []
        pred = []
        
        for batch_idx, batched_input in enumerate(valloader):
            
            # text_tensor, fine_tensor, coarse_tensor, true_coarses, true_fines, idx
            batch_size = len(batched_input[0])
            text_tensor = batched_input[0]
            fine_labels = batched_input[1]
            coarse_labels = batched_input[2]
            # the coarse labels that we know need
            labels = batched_input[3]
            ground_fines = batched_input[4]
            indexs = batched_input[-2]
            raw_texts = batched_input[-1]
            
            text_hidden, fine_hidden, coarse_hidden, logits, target, c_logits, c_target = model(batched_input[:4])
            loss, dist = criterion(text_hidden, fine_hidden, coarse_hidden, labels, ground_fines, logits, target, indexs, c_logits, c_target, epoch)
            for i in range(len(dist)):
                dist[i].extend([ground_fines[i].numpy().tolist(), indexs[i].numpy().tolist(), raw_texts[i]])
                sorted_pos = np.sort(dist[i][0])
                epoch_better.append([sorted_pos[-1], sorted_pos[-1] - sorted_pos[-2], labels[i].numpy().tolist(), indexs[i].numpy().tolist()])
            analyze_result.extend(dist)
            preds = prediction(text_hidden, fine_hidden, labels)
            pred.extend(preds)
            c_label.extend(labels.detach().cpu().numpy().tolist())
            f_label.extend(ground_fines.detach().cpu().numpy().tolist())
            
            if batch_idx % 100 == 0:
                print("Sample some true labeles and predicted labels")
                print(preds[:20])
                print(ground_fines[:20])

            acc = accuracy_score(np.array(preds), np.array(ground_fines.cpu()))
            macro_f1 = f1_score(np.array(preds), np.array(ground_fines.cpu()), average='macro')
            micro_f1 = f1_score(np.array(preds), np.array(ground_fines.cpu()), average='micro')
            correct += acc * labels.shape[0]
            macros += macro_f1 * labels.shape[0]
            micros += micro_f1 * labels.shape[0]
            loss_total += loss.item()
            total_sample += labels.shape[0]
        
        # we only select top args.adapter % confident instances 
        # sorted_pos[-1], sorted_pos[-1] - sorted_pos[-2], labels[i].numpy().tolist(), indexs[i].numpy().tolist()
        if args.select_mode == 'margin':
            # pdb.set_trace()
            if args.use_adapter:
                if args.use_coarse_label:
                    coarse_index = coarse_set.index(args.coarse_label)
                    epoch_better_new = [div[1] for div in epoch_better if div[2]==coarse_index]
                else:
                    epoch_better_new = [div[1] for div in epoch_better]
                sorted_bests = np.sort(epoch_better_new)
                select_size = int(len(epoch_better_new) * args.adapter)
                args.beta = sorted_bests[-select_size]
            else:
                args.beta = args.beta * 2   # if we dont want to use adapter, then we double the threshold
            # the indexs of selected confident instances
            # new_seed_index = [div[3] for div in epoch_better if div[0] >= args.beta]
            new_seed_index = [div[3] for div in epoch_better if div[1] >= args.beta]
            print('threshold margin is: ', args.beta)
        
        if args.select_mode == 'max':
            if args.use_adapter:
                if args.use_coarse_label:
                    coarse_index = coarse_set.index(args.coarse_label)
                    epoch_better_new = [div[0] for div in epoch_better if div[2]==coarse_index]
                else:
                    epoch_better_new = [div[0] for div in epoch_better]
                sorted_bests = np.sort(epoch_better_new)
                select_size = int(len(epoch_better_new) * args.adapter)
                args.maxbeta = sorted_bests[-select_size]
            else:
                args.maxbeta = args.maxbeta + 0.05   # if we dont want to use adapter, then 
            # the indexs of selected confident instances
            new_seed_index = [div[3] for div in epoch_better if div[0] >= args.maxbeta]
            print('threshold max is: ', args.maxbeta)
        
        # select the valid set (assume their prediction is their "label", since they are the most confident ones)
        # pdb.set_trace()
        # valid_text = np.array(text)[new_seed_index]
        valid_fine = np.array(f_label)[new_seed_index]
        # valid_coarse = np.array(c_label)[new_seed_index]
        valid_label = np.array(pred)[new_seed_index]
        
        
        # valid_dataset = PlainData(
        #     valid_text, valid_label, valid_coarse, tokenizer, 512)
            
        # calc the micro/macro F1 of the selected validation set
        valid_macro_f1 = f1_score(np.array(valid_label), np.array(valid_fine), average='macro')
        valid_micro_f1 = f1_score(np.array(valid_label), np.array(valid_fine), average='micro')
        
        print("FOR TEST ONLY==epoch {}, val macro {}, val micro {}".format(epoch, valid_macro_f1, valid_micro_f1))
        

        acc_total = correct/total_sample
        macro_total = f1_score(np.array(pred), np.array(f_label), average='macro')
        micro_total = f1_score(np.array(pred), np.array(f_label), average='micro')
        # macro_total = macros/total_sample
        # micro_total = micros/total_sample
        loss_total = loss_total/total_sample

        tmp_df['text'] = text
        tmp_df['coarse_label'] = c_label
        tmp_df['fine_label'] = f_label
        tmp_df['prediction'] = pred
        
        print("epoch {}, test loss avg. {}".format(epoch, loss_total))

    return loss_total, acc_total, macro_total, micro_total, tmp_df, new_seed_index, analyze_result, new_seed_index
    

def get_validate(valloader, model, epoch):
    analyze_result = []
    # store all the best - second best
    epoch_better = []
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        macros = 0
        micros = 0
        
        tmp_df = pd.DataFrame()
        text = []
        c_label = []
        f_label = []
        pred = []
        
        for batch_idx, batched_input in enumerate(valloader):
            
            # text_tensor, fine_tensor, coarse_tensor, true_coarses, true_fines, idx
            batch_size = len(batched_input[0])
            text_tensor = batched_input[0]
            fine_labels = batched_input[1]
            coarse_labels = batched_input[2]
            # the coarse labels that we know need
            labels = batched_input[3]
            ground_fines = batched_input[4]
            indexs = batched_input[-2]
            raw_texts = batched_input[-1]
            
            text_hidden, fine_hidden, coarse_hidden, logits, target, c_logits, c_target = model(batched_input[:4])
            # loss, dist = criterion(text_hidden, fine_hidden, coarse_hidden, labels, ground_fines, logits, target, indexs, c_logits, c_target, epoch)
            preds = prediction(text_hidden, fine_hidden, labels)
            pred.extend(preds)
            c_label.extend(labels.detach().cpu().numpy().tolist())
            f_label.extend(ground_fines.detach().cpu().numpy().tolist())
            
            if batch_idx % 100 == 0:
                print("Sample some true labeles and predicted labels")
                print(preds[:20])
                print(ground_fines[:20])

            acc = accuracy_score(np.array(preds), np.array(ground_fines.cpu()))
            macro_f1 = f1_score(np.array(preds), np.array(ground_fines.cpu()), average='macro')
            micro_f1 = f1_score(np.array(preds), np.array(ground_fines.cpu()), average='micro')
            correct += acc * labels.shape[0]
            macros += macro_f1 * labels.shape[0]
            micros += micro_f1 * labels.shape[0]
            # loss_total += loss.item()
            total_sample += labels.shape[0]
            
            print(total_sample)
            
        print(total_sample)
        if total_sample == 0:
            return 0.0, 0.0, 0.0
            
        acc_total = correct/total_sample
        # macro_total = macros/total_sample
        # micro_total = micros/total_sample
        macro_total = f1_score(np.array(pred), np.array(f_label), average='macro')
        micro_total = f1_score(np.array(pred), np.array(f_label), average='micro')
        # loss_total = loss_total/total_sample
            
    return acc_total, macro_total, micro_total
    



def get_data(data_path, use_coarse=False):
    global seed_index
    global weaksup_fine
    global weaksup_text
    global weaksup_coarse
    
    # Load the tokenizer for bert
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_seq_len = 512
    if args.weaksup_select == 'full':
        weaksup_df = pd.read_csv(os.path.join(data_path, dataset_name+'_weak_supervision.csv'))
    elif args.weaksup_select == 'coarse':
        weaksup_df = pd.read_csv(os.path.join(data_path, dataset_name+'_cweak_supervision.csv'))
    # weaksup_df = weaksup_df.sample(frac=1.0).reset_index(drop=True)
    train_df = pd.read_csv(os.path.join(data_path, args.dataset))
    # train_df = train_df.sample(frac=1.0).reset_index(drop=True)
    test_df = pd.read_csv(os.path.join(data_path, args.dataset))
    # test_df = test_df.sample(frac=1.0).reset_index(drop=True)
    train_df = convert_label_name(train_df)
    weaksup_df = convert_label_name(weaksup_df)
    test_df = convert_label_name(test_df)
    
    seed_index = select_initial_seed(train_df, weaksup_df)

    if use_coarse:
        train_df = train_df[train_df["coarse_label"].isin([args.coarse_label])].reset_index(drop=True)
        
    if use_coarse:
        test_df = test_df[test_df["coarse_label"].isin([args.coarse_label])].reset_index(drop=True)
    
    train_fine = []
    train_coarse = []
    train_text = []
    
    for i in range(len(train_df)):
        train_fine.append(train_df['fine_label'][i])
        train_text.append(train_df['text'][i])
        train_coarse.append(train_df['coarse_label'][i])
      
    # Here we only use the bodies and removed titles to do the classifications
    train_fine = np.array(train_fine)
    train_text = np.array(train_text)
    train_coarse = np.array(train_coarse)
    
    test_fine = []
    test_text = []
    test_coarse = []
    
    for i in range(len(test_df)):
        test_fine.append(test_df['fine_label'][i])
        test_text.append(test_df['text'][i])
        test_coarse.append(test_df['coarse_label'][i])

    test_fine = np.array(test_fine)
    test_text = np.array(test_text)
    test_coarse = np.array(test_coarse)
    
    for i in range(len(weaksup_df)):
        weaksup_fine.append(weaksup_df['fine_label'][i])
        weaksup_text.append(weaksup_df['text'][i])
        weaksup_coarse.append(weaksup_df['coarse_label'][i])

    weaksup_fine = np.array(weaksup_fine)
    weaksup_text = np.array(weaksup_text)
    weaksup_coarse = np.array(weaksup_coarse)

    # Build the dataset class for each set
    train_dataset = PlainData(
        train_text, train_fine, train_coarse, tokenizer, max_seq_len)
    test_dataset = PlainData(
        test_text, test_fine, test_coarse, tokenizer, max_seq_len)
    weaksup_dataset = PlainData(
        weaksup_text, weaksup_fine, weaksup_coarse, tokenizer, max_seq_len)
        
    print("#Train: {}, Test {}, Weaksup {}".format(len(
        train_coarse), len(test_coarse), len(weaksup_coarse)))

    return train_dataset, test_dataset, weaksup_dataset
    
    
def main():
    global best_acc
    global best_test_acc
    global weaksup_num

    # pdb.set_trace()
    
    # Read dataset and build dataloaders
    train_set, test_set, weaksup_set = get_data(args.data_dir)

    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    for data in train_set:
        print('?')
        break
    
    train_loader = Data.DataLoader(
        dataset=train_set, batch_size=8, shuffle=True)
    valid_loader = Data.DataLoader(
        dataset=weaksup_set, batch_size=128, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=128, shuffle=False)

    # Define the model, set the optimizer
    model = RobertaModelForLabelNCE().cuda()
    model = nn.DataParallel(model)
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  
                      eps=1e-8  
                      )
    
    
    seed_val = int(args.seed_val)

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    
    total_steps = len(train_loader) * args.epoch_num
    
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)


    test_accs = []
    test_macros = []
    test_micros = []
    
    val_accs = []
    val_macros = []
    val_micros = []
    
    val_full_metrics = []
    val_coarse_metrics = []
    val_fine_metrics = []
    
    best_macro_df = pd.DataFrame()
    best_micro_df = pd.DataFrame()
    label_stored = 0
    
    best_valid_macro = 0.0
    best_valid_micro = 0.0
    target_valid = 0

    pred_ground_labels = []
    
    print('AAAAAAA')
       
    # Start training
    for epoch in range(args.epoch_num):
    
        if epoch >= args.select_epoch and args.use_coarse_label:
            train_set, test_set, weaksup_set = get_data(args.data_dir)
            train_loader = Data.DataLoader(
                dataset=train_set, batch_size=8, shuffle=True)
            weaksup_loader = Data.DataLoader(
                dataset=weaksup_set, batch_size=128, shuffle=False)
            test_loader = Data.DataLoader(
                dataset=test_set, batch_size=128, shuffle=False)
        
        analyze_result = train(train_loader, model, optimizer,
             scheduler, epoch)
        
        valid_acc, valid_macro, valid_micro = get_validate(valid_loader, model, epoch)
        
        print("epoch {}, valid acc {}, valid macro_f1 {}, valid micro_f1 {}".format(
            epoch, valid_acc, valid_macro, valid_micro)) 
        
        test_loss, test_acc, test_macro, test_micro, tmp_test_df, new_seed_index, analyze_result, seed_index = validate(
            test_loader, model, epoch)
        test_df = pd.DataFrame(analyze_result, columns=['pos_dist', 'neg_dist', 'true_set', 'false_set', 'coarse_label', 'fine_label', 'index', 'raw_text'])
        test_df.to_csv('../result/' + ckpt_name + '_test_' + str(epoch) + '.csv')
        tmp_test_df['text'] = test_set.text
        
        # pdb.set_trace()
        valid_label = [fine_set[pred_label] for pred_label in tmp_test_df['prediction'].tolist()]
        valid_coarse = [coarse_set[pred_label] for pred_label in tmp_test_df['coarse_label'].tolist()]
        
        valid_text = np.array(tmp_test_df['text'].tolist())[new_seed_index]
        valid_label = np.array(valid_label)[new_seed_index]
        valid_coarse = np.array(valid_coarse)[new_seed_index]
        
        
        count = Counter(valid_coarse)
        print(count)
        
        valid_set = PlainData(valid_text, valid_label, valid_coarse, tokenizer, 512)
        
        # pdb.set_trace()
        
        print("epoch {}, test acc {}, test macro_f1 {}, test micro_f1 {}, test_loss {}".format(
            epoch, test_acc, test_macro, test_micro, test_loss))
            
        for data in valid_set:
            print('??')
            break
        
        valid_loader = Data.DataLoader(dataset=valid_set, batch_size=128, shuffle=False)
        
        for batched_data in valid_loader:
            print('aa')
            break

        if test_macro > best_valid_macro:
            best_valid_macro = test_macro
            torch.save(model, '../result/' + ckpt_name + '_best_macro.pth')
            best_macro_df = tmp_test_df
        
        if test_micro > best_valid_micro:
            best_valid_micro = test_micro
            torch.save(model, '../result/' + ckpt_name + '_best_micro.pth')
            best_micro_df = tmp_test_df
        
        
        if epoch < 5:
            torch.save(model, '../result/' + ckpt_name + str(epoch) + '.pth')
        else:
            if (epoch + 1) % 5 == 0:
                torch.save(model, '../result/' + ckpt_name + str(epoch) + '.pth')
        
        best_macro_df.to_csv('../result/' + ckpt_name + '_result.csv')
        best_micro_df.to_csv('../result/' + ckpt_name + '_result_micro.csv')

        
        
        test_accs.append(test_acc)
        val_accs.append(valid_acc)
        
        test_macros.append(test_macro)
        val_macros.append(valid_macro)
        
        test_micros.append(test_micro)
        val_micros.append(valid_micro)

        print("Finished training epoch", epoch)
        print('Best acc:')
        print(best_acc)
        
        print('Target epoch:')
        print(target_valid)
    
        print('Test acc:')
        print(test_accs)
        
        print('Test macro:')
        print(test_macros)
        
        print('Valid acc:')
        print(val_accs)
        
        print('Valid macro:')
        print(val_macros)
    
    with open(ckpt_name + '_epoch_result.json', 'w') as f:
        json.dump({'micros': test_accs, 'macros': test_macros, 'full_metric': val_full_metrics, 'coarse_metric': val_coarse_metrics, 'fine_metric': val_fine_metrics}, f)
    
    
    best_macro_df.to_csv('../result/' + ckpt_name + '_result.csv')
    best_micro_df.to_csv('../result/' + ckpt_name + '_result_micro.csv')



if __name__ == '__main__':
    main()