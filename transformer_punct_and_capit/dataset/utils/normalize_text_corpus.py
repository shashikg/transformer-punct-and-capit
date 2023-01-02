#!/usr/bin/env python
# coding: utf-8
# ----------------------------------------
# Maintainer: Shashi Kant 
# Userid: shashi.gupta
# ----------------------------------------

import os
import argparse
import pandas as pd

import ray
from ray import serve

import re, string
from tqdm import tqdm
from cleantext import clean
from .text_normalizers import load_normalizer
from ..utils import readTextData, writeTextData

import logging
logger = logging.getLogger("ray.serve")

num_workers = os.cpu_count()//8

class textFilter:
    def __init__(self, 
                 allowed_punct_marks=[".", "?", ","],
                 end_punct_marks=[".", "?"]):
            
        self.end_punct_marks = end_punct_marks
        self.allowed_punct_marks = "".join(allowed_punct_marks)
        
    def __call__(self, line):
        if len(line) == 0:
            return None
        else:
            cond = [
                line.upper() == line,
                line.lower() == line,
                re.sub(f'[{self.allowed_punct_marks}]', '', line) == line,
                line[-1] not in self.end_punct_marks,
                not line[0].isupper(),
            ]
            
            if any(cond):
                return None
            else:
                return line

re_multi_punct_remover = re.compile(r'([!*+,.:;<=>?_|~-])[!*+,.:;<=>?_|~-]+')
def clean_text(text):
    text = re.sub('-|~|\[.+?\]|\(.+?\)|\{.+?\}|\$|\^|\+|\=|\>|\<|\|', ' ', text)
    text = re.sub('¡|¿|»|«|"|…|—|\(|\)|\[|\]', ' ', text)
    text = clean(text, fix_unicode=True, to_ascii=False, lower=False, no_line_breaks=True, no_punct=False)
    text = re.sub('( )([!*+,.:;<=>?_|~-])', r'\2', text) # Remove space before punctuation
    text = re_multi_punct_remover.sub(r'\1', text) # Collapse multiple punctuation into single
    text = clean(text, fix_unicode=True, to_ascii=False, lower=False, no_line_breaks=True, no_punct=False)

    if (len(text) > 0) and text[0] == "'":
        text = text[1:]

    if (len(text) > 0) and text[-1] == "'":
        text = text[:-1]
    
    text = text.strip()

    return text

@serve.deployment(num_replicas=num_workers)
class NormalizationModel:
    def __init__(self, lang='en'):
        logger.setLevel(logging.ERROR)
        self.lang = lang
        self.normalizer = load_normalizer(self.lang)

    def __call__(self, text):
        try:
            text = self.normalizer(text)
        except:
            print("[Failed Norm]", text)
            text = ""
        
        return text
    
def normalize_text_data(handle, ip_fn, op_fn, punct_labels='O|.|,|?', end_punct_marks='.|?', buffer_size=1024):
    txt_corpus = readTextData(ip_fn)
    punct_labels = punct_labels.split("|")
    end_punct_marks = end_punct_marks.split("|")
    punct_labels.remove('O')

    filter_line = textFilter(allowed_punct_marks=punct_labels, end_punct_marks=end_punct_marks)
    
    norm_txt_corpus = []
    sent_need_normalisation = []
    for line in tqdm(txt_corpus, desc="[normalize_text_data]: Parsing data", total=len(txt_corpus)):
        if len(re.sub('[0123456789]', '', line)) != len(line):
            sent_need_normalisation.append(line)
        else:
            if len(line) > 1:
                norm_txt_corpus.append(line)
            
    print(f"[normalize_text_data]: Out of {len(txt_corpus):,} sentences only {len(sent_need_normalisation):,} need normalization and {len(norm_txt_corpus):,} does not need normalization.")
    
    non_finished = []
    for s_idx in tqdm(range(0, len(sent_need_normalisation), buffer_size), desc="[normalize_text_data]: Normalizing"):
        obj_refss = [handle.remote(x) for x in sent_need_normalisation[s_idx:s_idx+buffer_size]]
        
        for obj_ref in obj_refss:
            try:
                text = ray.get(obj_ref, timeout=1)
            except:
                non_finished.append(obj_ref)
                text = ""
                
            if len(text) > 1:
                norm_txt_corpus.append(text)
                
    finished, non_finished = ray.wait(non_finished, num_returns=len(non_finished), timeout=60)
    for obj_ref in finished:
        try:
            text = ray.get(obj_ref, timeout=1)
            text = filter_line(text)
            if text:
                text = clean_text(text)
            else:
                text = ""
        except:
            text = ""

        if len(text) > 1:
            norm_txt_corpus.append(text)
    
    writeTextData(op_fn, norm_txt_corpus)
    print(f"[normalize_text_data]: Total {len(norm_txt_corpus):,} number of lines.")
    