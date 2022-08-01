#!/usr/bin/env python
# coding: utf-8
# ----------------------------------------
# Maintainer: Shashi Kant 
# Userid: shashi.gupta
# ----------------------------------------

import os
import re
import argparse
import pandas as pd
from tqdm import tqdm

import ray
from ray import serve

from cleantext import clean
from .text_normalizers import load_normalizer
from ..utils import readTextData, writeTextData

import logging
logger = logging.getLogger("ray.serve")

num_workers = os.cpu_count()//8

@serve.deployment(num_replicas=num_workers)
class NormalizationModel:
    def __init__(self, lang='en'):
        logger.setLevel(logging.ERROR)
        self.lang = lang
#         print(f"Loading normalizer for lang: {self.lang}")
        self.normalizer = load_normalizer(self.lang)
        
    def clean_text(self, text, punct_labels):
        if "!" not in punct_labels:
            text = text.replace("!", ".")
        
        text = re.sub('-|~|\[.+?\]|\(.+?\)|\{.+?\}|\$|\^|\+|\=|\>|\<|\|', ' ', text)
        text = clean(text,fix_unicode=True,to_ascii=False,lower=False,no_line_breaks=True,no_punct=False)

        for _ in range(2):
            text = re.sub('¡|¿|»|«|"|…|—|\(|\)|\[|\]', ' ', text)
            text = text.replace("???", "?")
            text = text.replace("??", "?")
            text = text.replace("...", "")
            text = text.replace("..", ".")
            text = text.replace(",,,", ",")
            text = text.replace(",,", ",")

            text = text.replace("?.", "?")
            text = text.replace("?,", "?")

            text = text.replace(",.", ",")
            text = text.replace(",?", ",")

            text = text.replace(".?", ".")
            text = text.replace(".,", ".")

            text = text.replace(" ,", ",")
            text = text.replace(" .", ".")
            text = text.replace(" ?", "?")
            text = text.replace("' ", " ")
            text = text.replace(" '", " ")
            
            text = clean(text,fix_unicode=True,to_ascii=False,lower=False,no_line_breaks=True,no_punct=False)

        if (len(text) > 0) and text[0] == "'":
            text = text[1:]

        if (len(text) > 0) and text[-1] == "'":
            text = text[:-1]

        return text.strip()

    def __call__(self, x):
        text = x['text']
        punct_labels = x['punct_labels']
        
        try:
            text = self.normalizer(text)
            text = self.clean_text(text, punct_labels)
        except:
            print("[Failed Norm]", text)
            text = ""
        
        if len(text):
            cond = [
                "..." in text,
                ".." in text,
                "???" in text,
                "??" in text,
                ",,," in text,
                ",," in text,
                text.upper() == text,
                text.lower() == text,
                re.sub(f'[{"".join(punct_labels)}]', '', text) == text,
                text[0] == '?',
                text[0] == '.',
                text[0] == ',',
                ":" in text,
                not text[0].isupper(),
                text[-1] not in ['.', '?', '!', '|']
            ]
            if any(cond):
                text = ""
        
        return text

def clean_text(text, punct_labels):
    if "!" not in punct_labels:
        text = text.replace("!", ".")

    text = re.sub('-|~|\[.+?\]|\(.+?\)|\{.+?\}|\$|\^|\+|\=|\>|\<|\|', ' ', text)
    text = clean(text,fix_unicode=True,to_ascii=False,lower=False,no_line_breaks=True,no_punct=False)

    for _ in range(2):
        text = re.sub('¡|¿|»|«|"|…|—|\(|\)|\[|\]', ' ', text)
        text = text.replace("???", "?")
        text = text.replace("??", "?")
        text = text.replace("...", "")
        text = text.replace("..", ".")
        text = text.replace(",,,", ",")
        text = text.replace(",,", ",")

        text = text.replace("?.", "?")
        text = text.replace("?,", "?")

        text = text.replace(",.", ",")
        text = text.replace(",?", ",")

        text = text.replace(".?", ".")
        text = text.replace(".,", ".")

        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" ?", "?")
        text = text.replace("' ", " ")
        text = text.replace(" '", " ")

        text = clean(text,fix_unicode=True,to_ascii=False,lower=False,no_line_breaks=True,no_punct=False)

    if (len(text) > 0) and text[0] == "'":
        text = text[1:]

    if (len(text) > 0) and text[-1] == "'":
        text = text[:-1]
    
    text = text.strip()
    
    if len(text):
        cond = [
            "..." in text,
            ".." in text,
            "???" in text,
            "??" in text,
            ",,," in text,
            ",," in text,
            text.upper() == text,
            text.lower() == text,
            len(re.sub(f'[{"".join(punct_labels)}]', '', text)) == len(text),
            text[0] == '?',
            text[0] == '.',
            text[0] == ',',
            ":" in text,
            not text[0].isupper(),
            text[-1] not in ['.', '?', '!', '|']
        ]
        if any(cond):
            text = ""

    return text
    
def normalize_text_data(handle, ip_fn, op_fn, punct_labels='O|.|,|?', buffer_size=1024):
    txt_corpus = readTextData(ip_fn)
    punct_labels = punct_labels.split("|")
    punct_labels.remove('O')
    
    norm_txt_corpus = []
    sent_need_normalisation = []
    for i, line in tqdm(enumerate(txt_corpus), desc="[normalize_text_data]: Parsing data", total=len(txt_corpus)):
        if len(re.sub('[0123456789]', '', line)) != len(line):
            text = clean_text(line, punct_labels)
            if len(text) > 1:
                sent_need_normalisation.append(text)
        else:
            text = clean_text(line, punct_labels)
            if len(text) > 1:
                norm_txt_corpus.append(text)
            
    print(f"[normalize_text_data]: Out of {len(txt_corpus):,} sentences only {len(sent_need_normalisation):,} need normalization and {len(norm_txt_corpus):,} does not need normalization.")
    
    non_finished = []
    for s_idx in tqdm(range(0, len(sent_need_normalisation), buffer_size), desc="[normalize_text_data]: Normalizing"):
        obj_refss = [handle.remote({'text': x, 'punct_labels': punct_labels}) for x in sent_need_normalisation[s_idx:s_idx+buffer_size]]
        
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
        except:
            text = ""

        if len(text) > 1:
            norm_txt_corpus.append(text)
    
    writeTextData(op_fn, norm_txt_corpus)
    print(f"[normalize_text_data]: Total {len(norm_txt_corpus):,} number of lines.")
    