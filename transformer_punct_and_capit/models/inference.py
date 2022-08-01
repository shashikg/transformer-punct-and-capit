import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tarfile
import tempfile
import logging
import numpy as np
from omegaconf import OmegaConf

import torch
from torchcrf import CRF
from transformers import AutoTokenizer

from .utils import applyPunct
                   
class TransformerPunctAndCapitInferModel:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} does not exist")
        
        # Load model
        with tempfile.TemporaryDirectory() as tmpdir:
            tar = tarfile.open(model_path, "r:")
            tar.extractall(path=tmpdir)
            tar.close()
            
            config_path = os.path.join(tmpdir, "config.yaml")
            self.cfg = OmegaConf.load(config_path)
            self.cfg.tokenizer.path = str(os.path.join(tmpdir, "tokenizer"))
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer.path)
            
            jit_model_path = os.path.join(tmpdir, "jit_model.pt")
            self.jit_model = torch.jit.load(jit_model_path)
            
            if self.cfg.model.use_crf:
                crf_layer_path = os.path.join(tmpdir, "crf_layer.pt")
                crf_layer_state_dict = torch.load(crf_layer_path)
        
        if self.cfg.model.labels_dict is None:
            self.generate_labels_dict()
        
        # Configs
        self.pretrained_model_name = self.cfg.model.pretrained_model_name
        self.freeze_pretrained_model = self.cfg.model.freeze_pretrained_model
        self.use_crf = self.cfg.model.use_crf
        self.use_bilstm = self.cfg.model.use_bilstm
        self.labels_dict = self.cfg.model.labels_dict
        self.labels_order = self.cfg.model.labels_order.split("|")
        self.punct_labels = self.cfg.dataset.punct_labels.split("|")
        self.capit_labels = self.cfg.dataset.capit_labels.split("|")
        self.num_labels = len(self.labels_dict)
        
        # If CRF Layer
        if self.use_crf:
            self.crf_layer = CRF(self.num_labels, batch_first=True)
            self.crf_layer.load_state_dict(crf_layer_state_dict)
        
        # Tokenizers and metrics
        [START_ID, PAD_ID, END_ID, UNK_ID] = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token,
                                                                                   self.tokenizer.pad_token,
                                                                                   self.tokenizer.sep_token,
                                                                                   self.tokenizer.unk_token])
        self.token_style = {'START_SEQ': START_ID,
                            'PAD': PAD_ID,
                            'END_SEQ': END_ID,
                            'UNK': UNK_ID}
        
        self.apply_punct = applyPunct(self.labels_dict, self.labels_order)
        self.create_punct_helper()
        
        self.jit_model = self.jit_model.to(device).eval()
        self.crf_layer = self.crf_layer.to(device).eval()
        
    def create_punct_helper(self):
        capital_map = {}
        for key_1, value_1 in self.apply_punct.labels_dict.items():
            p_label_1 = key_1.split("|")[self.apply_punct.label_order_map['p']]
            c_label_1 = key_1.split("|")[self.apply_punct.label_order_map['c']]

            for key_2, value_2 in self.apply_punct.labels_dict.items():
                p_label_2 = key_2.split("|")[self.apply_punct.label_order_map['p']]
                c_label_2 = key_2.split("|")[self.apply_punct.label_order_map['c']]

                if p_label_1 == p_label_2 and c_label_2 == 'C':
                    if c_label_1 == 'U':
                        capital_map[value_1] = value_1
                    else:
                        capital_map[value_1] = value_2
                        
        self.capital_map = capital_map
        
        self.is_eos_map = {}
        for key, value in self.apply_punct.labels_dict.items():
            p_label = key.split("|")[self.apply_punct.label_order_map['p']]

            if p_label in ['.', '?', '!', '|']:
                self.is_eos_map[value] = True
            else:
                self.is_eos_map[value] = False

    def generate_labels_dict(self):
        idx = 0
        labels_dict = {}
        labels_order = self.cfg.dataset.labels_order.split("|")
        for punct in self.cfg.dataset.punct_labels.split("|"):
            for capit in self.cfg.dataset.capit_labels.split("|"):
                key = {'p': punct, 'c': capit}
                labels_dict[f"{key[labels_order[0]]}|{key[labels_order[1]]}"] = idx
                idx += 1

        self.cfg.model.labels_dict = labels_dict
        self.cfg.model.labels_order = self.cfg.dataset.labels_order
    
    @torch.no_grad()
    def forward(self, x, attn_masks):
        """
        Forward pass of the model.
        """
        
        x, attn_masks = x.to(self.device), attn_masks.to(self.device)
        y_pred = torch.zeros(x.shape).long().to(self.device)
        x = self.jit_model(x, attn_masks)
        
        if self.use_crf:
            attn_masks = attn_masks.byte()
            dec_out = self.crf_layer.decode(x, mask=attn_masks)
            for i in range(len(dec_out)):
                y_pred[i, :len(dec_out[i])] = torch.tensor(dec_out[i]).to(self.device)
        else:
            y_pred = torch.argmax(x, dim=-1)
            
        return y_pred
    
    def tokenize_by_batch(self, queries):
        """
        Prepare tokens for input sentences. Combining all sentence and performing tokenization works faster than doing them serially.
        
        Inputs:
            - queries: List of sentences
        
        Ouputs:
            - st: subtokens
            - stm: subtoken mask 
            - seq_lengths: numbers of tokens in each sentences
        """
        
        st = []
        stm = []
        seq_lengths = []

        subtokens_all = self.tokenizer.tokenize(f" {self.tokenizer.sep_token} ".join(queries).strip())
        subtokens = []
        subtokens_mask = []
        for word_tokens in subtokens_all:
            if word_tokens == self.tokenizer.sep_token:
                subtokens_mask = subtokens_mask[1:] + [True] # Shift one place so that True value gets at the end of each word.
                seq_lengths.append(len(subtokens))
                st.append(subtokens)
                stm.append(subtokens_mask)
                subtokens = []
                subtokens_mask = []
            elif (word_tokens[0] != "#") and (word_tokens[0] != "'") and ((len(subtokens) == 0) or (subtokens[-1] != "'")):
                # condition to detect start of word
                subtokens.append(word_tokens)
                subtokens_mask.append(True)
            else:
                subtokens.append(word_tokens)
                subtokens_mask.append(False)
        
        subtokens_mask = subtokens_mask[1:] + [True]
        seq_lengths.append(len(subtokens))
        st.append(subtokens)
        stm.append(subtokens_mask)

        return st, stm, seq_lengths
    
    def prep_batch(self, queries, max_seq_length):
        """
        Prepare input sentences to feed in BERT model.
        
        Inputs:
            - queries: List of sentences
        
        Ouputs:
            - x: input tokens
            - y_masks: input token mask 
            - attn_masks: to define padded and original token segments in input_token
            - query_ids: List of ids corresponding to the sentence number in the original queries list
        """
        
        all_subtokens, all_subtokens_mask, all_subtokens_lengths = self.tokenize_by_batch(queries)

        min_max_seq_length = max(all_subtokens_lengths) + 2
        if max_seq_length > min_max_seq_length:
            max_seq_length = min_max_seq_length

        length = max_seq_length - 2

        x, y_masks, attn_masks, query_ids = [], [], [], []

        for q_id in range(len(all_subtokens)):
            q_subtokens, q_subtokens_mask, q_subtokens_length = all_subtokens[q_id], all_subtokens_mask[q_id], all_subtokens_lengths[q_id]

            for i in range(0, q_subtokens_length, length):
                if (i+length) >= q_subtokens_length:
                    q_x = [self.token_style['START_SEQ']] + self.tokenizer.convert_tokens_to_ids(q_subtokens[i:]) + [self.token_style['END_SEQ']] + (i+length-q_subtokens_length)*[self.token_style['PAD']]
                    q_y_masks = [False] + q_subtokens_mask[i:] + [False] + (i+length-q_subtokens_length)*[False]
                    q_attn_masks = [1]*(q_subtokens_length + 2 - i) + (i+length-q_subtokens_length)*[0]
                else:
                    q_x = [self.token_style['START_SEQ']] + self.tokenizer.convert_tokens_to_ids(q_subtokens[i:i+length]) + [self.token_style['END_SEQ']]
                    q_y_masks = [False] + q_subtokens_mask[i:i+length] + [False]
                    q_attn_masks = [1]*max_seq_length

                x.append(q_x)
                y_masks.append(q_y_masks)
                attn_masks.append(q_attn_masks)
                query_ids.append(q_id)

        return x, y_masks, attn_masks, query_ids
    
    @torch.no_grad()
    def predict_batch(self, queries_all, batch_size=None, max_seq_length=64):
        """
        Performs inference and applies punct. and capit.
        
        Inputs:
            - queries_all: List of sentences
            - batch_size: if custom batch size given uses that otherwise uses batch_size = len(queries_all)
        
        Ouputs:
            - resp_texts: List of punctuated texts
        """
        
        if batch_size == None:
            batch_size = len(queries_all)
            
        try:
            resp_texts = []
            for idx_1 in range(0, len(queries_all), batch_size):
                queries = queries_all[idx_1:idx_1+batch_size]

                x, y_masks, attn_masks, query_ids = self.prep_batch(queries, max_seq_length=max_seq_length)

                x = torch.tensor(x, device=self.device)
                attn_masks = torch.tensor(attn_masks, device=self.device)

                punct_pred = self.forward(x, attn_masks)
                punct_pred = punct_pred.cpu().numpy()

                lines = [line.strip().split() for line in queries]
                p_resp = [[] for _ in range(len(queries))]

                for idx_2, q_id in enumerate(query_ids):
                    p_resp[q_id] += list(punct_pred[idx_2][y_masks[idx_2]])
                
                for q_id in query_ids:
                    if len(p_resp[q_id]) > 1:
                        p_prev_label = p_resp[q_id][1]
                        
                        for idx_2 in range(2, len(p_resp[q_id]), 1):
                            p_curr_label = p_resp[q_id][idx_2]
                            
                            if self.is_eos_map[p_prev_label]:
                                p_resp[q_id][idx_2] = self.capital_map[p_curr_label]

                            p_prev_label = p_curr_label

                for req_id in range(len(lines)):
                    resp_texts.append(" ".join([self.apply_punct(word, pl) for word, pl in zip(lines[req_id], p_resp[req_id])]))
            
            return resp_texts, True
        
        except Exception as ex:
            logging.exception("exception in punctuation:{}".format(ex))
            return queries_all, False
        
        
        
        
        
        

                                 

