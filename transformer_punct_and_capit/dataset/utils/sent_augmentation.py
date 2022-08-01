import os
import re
import copy
import numpy as np
from tqdm import tqdm
from ..utils import readTextData, writeTextData

def combine_sentences(txt_corpus, combine_ratio=0.66):
    idx_with_one_sent = []
    idx_not_to_combine = []
    for idx, line in tqdm(enumerate(txt_corpus), total=len(txt_corpus), desc="[sent_augmentation]: parsing data"):
        num_sent = line.count("?")+line.count(".")
        if (num_sent == 1) and (len(line) <= 80):
            idx_with_one_sent.append(idx)
        else:
            idx_not_to_combine.append(idx)

    num_sent_to_combine = int(combine_ratio*len(idx_with_one_sent))
    num_sent_to_combine = num_sent_to_combine if num_sent_to_combine%2 == 0 else num_sent_to_combine -1

    np.random.shuffle(idx_with_one_sent)
    idx_to_combine = idx_with_one_sent[:num_sent_to_combine]
    idx_not_to_combine += idx_with_one_sent[num_sent_to_combine:]
    
    merged_idx_list = []
    sent_len = np.random.choice([2, 3, 4, 5])
    curr_idx_list = []
    for i in range(len(idx_to_combine)):
        curr_idx_list.append(i)

        if len(curr_idx_list) == sent_len:
            sent_len = np.random.choice([2, 3, 4, 5])
            merged_idx_list.append(curr_idx_list)
            curr_idx_list = []

    new_txt_corpus = []
    for curr_idx_list in tqdm(merged_idx_list, desc=f"[sent_augmentation]: combining sentences"):
        tmp_txt_list = [txt_corpus[idx_to_combine[i]] for i in curr_idx_list]
        new_txt_corpus.append(" ".join(tmp_txt_list))

    for idx in idx_not_to_combine:
        new_txt_corpus.append(txt_corpus[idx])

    np.random.shuffle(new_txt_corpus)

    return new_txt_corpus

def cut_sentences(txt_corpus, cut_ratio=0.2):
    random_idx_seq = [*range(len(txt_corpus))]
    np.random.shuffle(random_idx_seq)

    num_sent_to_cut = int(len(random_idx_seq)*cut_ratio)
    idx_to_cut = random_idx_seq[:num_sent_to_cut]
    idx_not_to_cut = random_idx_seq[num_sent_to_cut:]

    new_txt_corpus = []

    for idx in tqdm(idx_to_cut, f"[sent_augmentation]: cutting sentences"):
        line = txt_corpus[idx]
        line = line.split()
        sent1 = ' '.join(line[:(len(line)//2)])
        sent2 = ' '.join(line[(len(line)//2):])

        if len(sent1) > 1:
            new_txt_corpus.append(copy.deepcopy(sent1))

        if len(sent2) > 1:
            new_txt_corpus.append(copy.deepcopy(sent2))

    for idx in idx_not_to_cut:
        new_txt_corpus.append(txt_corpus[idx])

    np.random.shuffle(new_txt_corpus)

    return new_txt_corpus

def apply_all(ip_fn, op_fn, combine_ratio=0.66, cut_ratio=0.2):
    txt_corpus = readTextData(ip_fn)
    txt_corpus = combine_sentences(txt_corpus, combine_ratio=combine_ratio)
    txt_corpus = cut_sentences(txt_corpus, cut_ratio=cut_ratio)
    writeTextData(op_fn, txt_corpus)
    print(f"[sent_augmentation]: Total {len(txt_corpus):,} number of lines.")