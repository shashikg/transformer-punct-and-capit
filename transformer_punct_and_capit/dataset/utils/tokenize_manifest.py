import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import ray
from ray.util import ActorPool

import msgpack
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from ..utils import readJsonManifest, savePackedData

@ray.remote
class TokenizerActor:
    def __init__(self, tokenizer, labels_dict, max_seq_length):
        self.tokenizer = tokenizer
        self.labels_dict = labels_dict
        self.max_seq_length = max_seq_length
        
        [self.START_ID, self.PAD_ID, self.END_ID, self.UNK_ID] = tokenizer.convert_tokens_to_ids([tokenizer.cls_token,
                                                                                                  tokenizer.pad_token,
                                                                                                  tokenizer.sep_token,
                                                                                                  tokenizer.unk_token])
        
    def tokenize_sample(self, sample):
        lines = []
        for w, p in zip(sample['x_text'].split(" "), sample['labels'].split(" ")):
            if len(w) != 0:
                lines.append((w, p))

        idx = 0
        data_items = []
        while idx < len(lines): # loop until end of the sentence
            x = [self.START_ID]
            y = [0]
            y_mask = [1]  # which positions we need to consider while evaluating i.e., ignore pad or sub tokens

            # loop until we have required sequence length
            # -1 because we will have a special end of sequence token at the end
            while len(x) < self.max_seq_length - 1 and idx < len(lines):
                word, punct = lines[idx]
                tokens = self.tokenizer.tokenize(word)

                # if taking these tokens exceeds sequence length we finish current sequence with padding
                # then start next sequence from this token
                if len(tokens) + len(x) >= self.max_seq_length:
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x.append(self.tokenizer.convert_tokens_to_ids(tokens[i]))
                        y.append(0)
                        y_mask.append(0)

                    if len(tokens) > 0:
                        x.append(self.tokenizer.convert_tokens_to_ids(tokens[-1]))
                    else:
                        x.append(self.UNK_ID)

                    y.append(self.labels_dict[punct])
                    y_mask.append(1)
                    idx += 1

            x.append(self.END_ID)
            y.append(0)
            y_mask.append(1)

            if len(x) < self.max_seq_length:
                x += [self.PAD_ID]*(self.max_seq_length - len(x))
                y += [0]*(self.max_seq_length - len(y))
                y_mask += [0]*(self.max_seq_length - len(y_mask))

            attn_mask = [1 if token != self.PAD_ID else 0 for token in x]
            data_items.append([x, y, attn_mask, y_mask])

        return data_items
    
def tokenize_manifest(man_fn, tokenizer_path, labels_dict, max_seq_length, 
                      save_data=True, num_worker=64):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer_actors = [TokenizerActor.remote(tokenizer, labels_dict, max_seq_length) for _ in range(num_worker)]
    pool = ActorPool(tokenizer_actors)
    
    data = readJsonManifest(man_fn)
    data_items = []
    pbar_desc = f"[tokenize_manifest]: Tokenizing"
    for _ in tqdm(pool.map(lambda tokenizer_actor, sample: tokenizer_actor.tokenize_sample.remote(sample), data), total=len(data), desc=pbar_desc):
        data_items += _

    for _ in tokenizer_actors:
        ray.kill(_)
    
    if save_data:
        out_fn = man_fn.replace(".json", f"_{max_seq_length}.msgpack")
        savePackedData(out_fn, data_items)
        return out_fn
    else:
        return data_items