import os
import numpy as np
import msgpack
import json
from tqdm import tqdm
import rich.progress

def savePackedData(fn, data):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "wb") as outfile:
        packed = msgpack.packb(data)
        outfile.write(packed)

def loadPackedData(fn):
    with rich.progress.open(fn, "rb") as data_file:
        byte_data = data_file.read()
    
    data = msgpack.unpackb(byte_data)
    return data

def writeTextData(fn, txt_corpus):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    
    with open(fn, 'w') as fout:
        for line in tqdm(txt_corpus, desc='writing'):
            fout.write(line + '\n')

def readTextData(fn):
    txt_corpus = []
    with rich.progress.open(fn, "r") as file:
        for line in file:
            line = line.split("\n")[0].strip()
            if line:
                txt_corpus.append(line)
                
    return txt_corpus

def readJsonManifest(man_fn):
    data = []
    with rich.progress.open(man_fn, "r") as fio:
        for line in fio:
            sample = json.loads(line)
            data.append(sample)
        
    return data

def merge_multiple_text_data(text_files, max_lines=-1):
    merged_corpus = []
    
    for fn in tqdm(text_files, desc="[merge_multiple_text_data]: merging"):
        tmp_corpus = readTextData(fn)
        
        if max_lines>0:
            np.random.shuffle(tmp_corpus)
            tmp_corpus = tmp_corpus[:max_lines]
            
        merged_corpus += tmp_corpus
        
    merged_corpus = list(set(merged_corpus))
    np.random.shuffle(merged_corpus)
    
    return merged_corpus

def generate_labels_dict(cfg):
    idx = 0
    labels_dict = {}
    labels_order = cfg.dataset.labels_order.split("|")
    for punct in cfg.dataset.punct_labels.split("|"):
        for capit in cfg.dataset.capit_labels.split("|"):
            key = {'p': punct, 'c': capit}
            labels_dict[f"{key[labels_order[0]]}|{key[labels_order[1]]}"] = idx
            idx += 1

    cfg.model.labels_dict = labels_dict
    cfg.model.labels_order = cfg.dataset.labels_order
    return cfg
    
    