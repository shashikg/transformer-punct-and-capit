import numpy as np
from tqdm import tqdm
from ..utils import readTextData, writeTextData

def balance_text_data(ip_fn, op_fn, punct_labels='O|.|,|?', balancing_margin_ratio=10):
    txt_corpus = readTextData(ip_fn)
    len_orig = len(txt_corpus)

    txt_corpus = list(set(txt_corpus))
    len_after_removing_duplicates = len(txt_corpus)

    print(f"[balance_text_data]: Original data length: {len_orig:,} - After removing duplicates: {len_after_removing_duplicates:,}")
    
    punct_labels = punct_labels.split("|")
    punct_labels.remove('O')
    punct_labels = np.array(punct_labels)
    
    punct_count = []
    for i, line in tqdm(enumerate(txt_corpus), desc="[balance_text_data]: Count punctuations", total=len(txt_corpus)):
        tmp_cnt = []
        for p in punct_labels:
            tmp_cnt.append(line.count(p))
            
        punct_count.append(tmp_cnt)
    punct_count = np.array(punct_count)
    
    current_count = np.sum(punct_count, axis=0)
    print(f"[balance_text_data]: Dataset initially have | {current_count[0]:,} - ({punct_labels[0]}) | {current_count[1]:,} - ({punct_labels[1]}) | {current_count[2]:,} - ({punct_labels[2]}) |")
    punct_sorted_idx = np.argsort(np.sum(punct_count, axis=0))
    
    final_list = []
    for i in range(len(punct_sorted_idx)):
        logit = punct_count[:, punct_sorted_idx[i]]!=0
        for j in range(i-1, -1, -1):
            logit = logit & (punct_count[:, punct_sorted_idx[j]]==0)
        
        current_count = np.sum(punct_count[final_list], axis=0)
        
        if i == 0:
            final_list += list(np.argwhere(logit).reshape(-1))
        elif current_count[punct_sorted_idx[i]] < balancing_margin_ratio*current_count[punct_sorted_idx[0]]:
            curr_punct_sent = list(np.argwhere(logit).reshape(-1))
            np.random.shuffle(curr_punct_sent)
            punct2sent_factor = np.sum(punct_count[curr_punct_sent], axis=0)[punct_sorted_idx[i]]/len(curr_punct_sent)
            num_sent_to_take = balancing_margin_ratio*current_count[punct_sorted_idx[0]] - current_count[punct_sorted_idx[i]]
            num_sent_to_take = int(num_sent_to_take/punct2sent_factor)
            final_list += curr_punct_sent[:num_sent_to_take]
            
    current_count = np.sum(punct_count[final_list], axis=0)
    
    np.random.shuffle(final_list)
    balanced_txt_corpus = []
    for idx in final_list:
        balanced_txt_corpus.append(txt_corpus[idx])

    writeTextData(op_fn, balanced_txt_corpus)
    print(f"[balance_text_data]: Total {len(balanced_txt_corpus):,} number of lines.")
    print(f"[balance_text_data]: Having | {current_count[0]:,} - ({punct_labels[0]}) | {current_count[1]:,} - ({punct_labels[1]}) | {current_count[2]:,} - ({punct_labels[2]}) |")
