import os
import numpy as np
from tqdm import tqdm
from ..utils import readTextData, writeTextData

def split_text_data(ip_fn, save_dir, split_ratio=0.03):
    os.makedirs(save_dir, exist_ok=True)
    
    txt_corpus = readTextData(ip_fn)
    txt_corpus = list(set(txt_corpus))
    np.random.shuffle(txt_corpus)

    train_fn = f'{save_dir}/{ip_fn.split("/")[-1].replace(".txt", "_train.txt")}'
    dev_fn = f'{save_dir}/{ip_fn.split("/")[-1].replace(".txt", "_dev.txt")}'
    test_fn = f'{save_dir}/{ip_fn.split("/")[-1].replace(".txt", "_test.txt")}'
    
    num_samples = int(len(txt_corpus)*split_ratio)

    writeTextData(train_fn, txt_corpus[:-2*num_samples])
    writeTextData(dev_fn, txt_corpus[-2*num_samples:-num_samples])
    writeTextData(test_fn, txt_corpus[-num_samples:])
    
    print(f"[split_text_data]: Train split total {len(txt_corpus[:-2*num_samples]):,} number of lines.")
    print(f"[split_text_data]: Dev split total {num_samples:,} number of lines.")
    print(f"[split_text_data]: Test split total {num_samples:,} number of lines.")
