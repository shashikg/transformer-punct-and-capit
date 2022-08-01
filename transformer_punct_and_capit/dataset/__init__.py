import os
import torch
from .augmentation import AUGMENTATIONS
import numpy as np
from tqdm import tqdm
from .utils import loadPackedData

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, sequence_len, is_train=False, 
                 augmentation_args=None):
        
        [START_ID, PAD_ID, END_ID, UNK_ID] = tokenizer.convert_tokens_to_ids([tokenizer.cls_token,
                                                                              tokenizer.pad_token,
                                                                              tokenizer.sep_token,
                                                                              tokenizer.unk_token])
        self.token_style = {'START_SEQ': START_ID,
                            'PAD': PAD_ID,
                            'END_SEQ': END_ID,
                            'UNK': UNK_ID}

        self.data = loadPackedData(data_path)
        self.sequence_len = sequence_len
        self.is_train = is_train
        self.augmentation_args = augmentation_args

    def __len__(self):
        return len(self.data)

    def _augment(self, x, y, y_mask):
        x_aug = []
        y_aug = []
        y_mask_aug = []
        for i in range(len(x)):
            r = np.random.rand()
            if self.augmentation_args and r < self.augmentation_args['rate']:
                AUGMENTATIONS[self.augmentation_args['type']](x, y, y_mask, x_aug, y_aug, y_mask_aug, i, 
                                                              self.token_style, **self.augmentation_args)
            else:
                x_aug.append(x[i])
                y_aug.append(y[i])
                y_mask_aug.append(y_mask[i])

        if len(x_aug) > self.sequence_len:
            # len increased due to insert
            x_aug = x_aug[0:self.sequence_len]
            y_aug = y_aug[0:self.sequence_len]
            y_mask_aug = y_mask_aug[0:self.sequence_len]
        elif len(x_aug) < self.sequence_len:
            # len decreased due to delete
            x_aug = x_aug + [self.token_style['PAD'] for _ in range(self.sequence_len - len(x_aug))]
            y_aug = y_aug + [0 for _ in range(self.sequence_len - len(y_aug))]
            y_mask_aug = y_mask_aug + [0 for _ in range(self.sequence_len - len(y_mask_aug))]

        attn_mask = [1 if token != self.token_style['PAD'] else 0 for token in x]
        return x_aug, y_aug, attn_mask, y_mask_aug

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        attn_mask = self.data[index][2]
        y_mask = self.data[index][3]

        if self.is_train and self.augmentation_args and self.augmentation_args['rate'] > 0:
            x, y, attn_mask, y_mask = self._augment(x, y, y_mask)

        x = torch.tensor(x)
        y = torch.tensor(y)
        attn_mask = torch.tensor(attn_mask)
        y_mask = torch.tensor(y_mask)

        return x, y, attn_mask, y_mask

class InferDataset(torch.utils.data.Dataset):
    def __init__(self, data_items):
        self.data = data_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sent_idx = self.data[index][0]
        x = self.data[index][1]
        attn_mask = self.data[index][2]
        y_mask = self.data[index][3]

        x = torch.tensor(x)
        attn_mask = torch.tensor(attn_mask)
        y_mask = torch.tensor(y_mask)
        
        return sent_idx, x, attn_mask, y_mask
