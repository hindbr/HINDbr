import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from modules import extract_bug_report_information
from itertools import combinations
import json
import pickle
import sys

import os

'''
Generate an example of hin, the result is used to draw graph through Cytoscape.

'''


# Project Setting
PROJECT = 'gnome'


BUG_GROUP_FNAME = 'data/bug_report_groups/' + PROJECT + '_all.pkl'

with open('data/xmlfile_path.setting') as f:
    XML_FILE_PATH_DICT = json.load(f)

XML_FILE_PATH = XML_FILE_PATH_DICT[PROJECT][0]
XML_FILE_PREFIX = XML_FILE_PATH_DICT[PROJECT][1]

HIN_EMBEDDING_DIM = '128'
HIN_EMBEDDING_FILE = 'data/pretrained_embeddings/hin2vec/' + PROJECT + '_node_' + HIN_EMBEDDING_DIM + 'd_5n_4w_1280l.vec'
HIN_NODE_DICT = 'data/hin_node_dict/' + PROJECT + '_node.dict'



with open(BUG_GROUP_FNAME, 'rb') as f:
    bug_bucket = pickle.load(f)

duplicate_pairs = list()
master = 572385 
duplicates = bug_bucket[master]
duplicates.add(master)
for pair in combinations(duplicates, 2):
    duplicate_pairs.append(pair)
all_bugs=duplicates
all_bugs.add(381618)
non_dup_pairs = list()
for bug in all_bugs:
    if bug != 381618:
        non_dup_pairs.append((381618,bug))

print(duplicate_pairs)
print(non_dup_pairs)

bug_dict = dict()
for bug in all_bugs:
    information1 = extract_bug_report_information(XML_FILE_PATH + XML_FILE_PREFIX + str(bug) + '.xml')
    summary1 = information1[0]
    description1 = information1[1]
    pro1 = information1[2]
    com1 = information1[3]
    ver1 = information1[4]
    sev1 = information1[5]
    pri1 = information1[6]

    bug_dict[bug] = [bug, pro1, com1, ver1, sev1, pri1]

#["bid1","pro1","com1","ver1","sev1","pri1"]
#R1 BID-COM
#R2 COM-PRO
#R3 BID-VER
#R4 BID-PRI
#R5 BID-SEV
SAVE_HIN_FILE = 'output/example_of_hin/gnome_master_572385_dup_hin.txt'
SAVE_HIN_NODE_TYPE = 'output/example_of_hin/gnome_master_572385_dup_hin_node_type.txt'
SAVE_HIN_SIMI = 'output/example_of_hin/gnome_master_pair_similarities.txt'

f_hin = open(SAVE_HIN_FILE, 'w')
f_nt = open(SAVE_HIN_NODE_TYPE, 'w')
f_sim = open(SAVE_HIN_SIMI,'w')
bug_hin_vector_dict = dict()
hin_dict = dict()
hin_node_type_dict = dict()
# Load hin node2vec
node2vec = {}
with open(HIN_EMBEDDING_FILE) as f:
    first = True
    for line in f:
        if first:
            first = False
            continue
        line = line.strip()
        tokens = line.split(' ')
        node2vec[tokens[0]] = np.array(tokens[1:],dtype=float)

with open(HIN_NODE_DICT, 'r') as f:
    hin_node_dict = json.load(f)




for bug in bug_dict:
    bid = bug_dict[bug][0]
    pro = bug_dict[bug][1]
    com = bug_dict[bug][2]
    ver = bug_dict[bug][3]
    sev = bug_dict[bug][4]
    pri = bug_dict[bug][5]
    if str(bid)+com not in hin_dict:
        hin_dict[str(bid)+com] = [str(bid),'R1',str(com)]
    if com+pro not in hin_dict:
        hin_dict[com+pro] = [str(com),'R2',str(pro)]
    if str(bid)+ver not in hin_dict:
        hin_dict[str(bid)+ver] = [str(bid),'R3',str(ver)]
    if str(bid)+pri not in hin_dict:
        hin_dict[str(bid)+pri] = [str(bid),'R4',str(pri)]
    if str(bid)+sev not in hin_dict: 
        hin_dict[str(bid)+sev] = [str(bid),'R5',str(sev)]
    if bid not in hin_node_type_dict:
        hin_node_type_dict[bid] = 'BID'
    if pro not in hin_node_type_dict:
        hin_node_type_dict[pro] = 'PRO'
    if com not in hin_node_type_dict:
        hin_node_type_dict[com] = 'COM'
    if ver not in hin_node_type_dict:
        hin_node_type_dict[ver] = 'VER'
    if sev not in hin_node_type_dict:
        hin_node_type_dict[sev] = 'SEV'
    if pri not in hin_node_type_dict:
        hin_node_type_dict[pri] = 'PRI'

    bug_hin_vector_dict[bug] = np.concatenate([node2vec[str(hin_node_dict[str(bid)][0])],node2vec[str(hin_node_dict[str(pro)][0])],node2vec[str(hin_node_dict[str(com)][0])],node2vec[str(hin_node_dict[str(ver)][0])],node2vec[str(hin_node_dict[str(sev)][0])],node2vec[str(hin_node_dict[str(pri)][0])]]) 

def manhattan_distance_similarity(v1,v2):
    return np.exp(-np.sum(np.abs(v1-v2)))    

pair_similarity = dict()
for pair in duplicate_pairs:
    pair_similarity[pair] = manhattan_distance_similarity(bug_hin_vector_dict[pair[0]],bug_hin_vector_dict[pair[1]])

for pair in non_dup_pairs:
    pair_similarity[pair] = manhattan_distance_similarity(bug_hin_vector_dict[pair[0]],bug_hin_vector_dict[pair[1]])

new_pair_similarity = dict()

for key in pair_similarity:
    new_pair_similarity[key] = (pair_similarity[key]-min(pair_similarity[key] for key in pair_similarity))/(max(pair_similarity[key] for key in pair_similarity)-min(pair_similarity[key] for key in pair_similarity))


for key in hin_dict:
    f_hin.write(hin_dict[key][0] + '\t' + hin_dict[key][1] + '\t' + hin_dict[key][2] + '\n')
for key in hin_node_type_dict:
    f_nt.write(str(key) + '\t' + hin_node_type_dict[key] + '\n')
for key in new_pair_similarity:
    f_sim.write(str(key) + '\t' + str(new_pair_similarity[key]) + '\n')
f_hin.close()
f_nt.close()
f_sim.close()
