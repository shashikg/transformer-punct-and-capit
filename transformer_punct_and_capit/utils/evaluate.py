import re
import copy
import string
import numpy as np
from cleantext import clean

from rich import box
from rich.table import Table
from rich.console import Console
console = Console()

import torch
import torchmetrics
from tqdm import tqdm
from ..metric import ClassificationStatScores

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

class evaluatePredictions:
    def __init__(self, cfg):
        self.labels_dict = cfg.model.labels_dict
        self.labels_order = cfg.model.labels_order.split("|")
        self.punct_labels = cfg.dataset.punct_labels.split("|")
        self.capit_labels = cfg.dataset.capit_labels.split("|")
        self.num_labels = len(self.labels_dict)
        
        # New labels mapping
        self.indv_label_map = {
            'c': {k: v for v, k in enumerate(self.capit_labels)},
            'p': {k: v for v, k in enumerate(self.punct_labels)}
        }
        
        # Map model output labels to different punct and capit case
        self.label_map = {'c': {}, 'p': {}} 
        for key, value in self.labels_dict.items():
            for i in range(2):
                try:
                    self.label_map[self.labels_order[i]][key.split("|")[i]].append(value)
                except:
                    self.label_map[self.labels_order[i]][key.split("|")[i]] = [value]
        
        # Setup val metrics
        self.val_metrics = {}
        for _ in self.labels_order:
            self.val_metrics[_] = {
                'label_wise': {
                    'stats': ClassificationStatScores(num_classes=len(self.label_map[_]), average='none'),
                    'Accuracy': torchmetrics.Accuracy(num_classes=len(self.label_map[_]), average='none'),
                },
                'macro_average': {
                    'stats': ClassificationStatScores(num_classes=len(self.label_map[_]), average='macro'),
                    'Accuracy': torchmetrics.Accuracy(num_classes=len(self.label_map[_]), average='macro'),
                }
            }
            
        self.val_metrics['All'] = {
            'macro_average': {
                'stats': ClassificationStatScores(num_classes=self.num_labels, average='macro'),
                'Accuracy': torchmetrics.Accuracy(num_classes=self.num_labels, average='macro'),
            }
        }
        
        self.header_map = {'p': 'Punct', 'c': 'Capit', 'All': 'All'}
        
    def clean_text(self, text):
        return clean_text(text)

    def get_word_label(self, word):
        label = {}
        label['p'] = word[-1] if word[-1] in self.punct_labels else 'O'
        
        if 'U' in self.capit_labels:
            upper_case_sym = 'U'
        else:
            upper_case_sym = 'C'

        if len(word) > 0:
            if word.isupper():
                label['c'] = upper_case_sym
            elif word[0].isupper():
                label['c'] = 'C'
            else:
                label['c'] = 'O'
        else:
            label['c'] = 'O'

        return self.labels_dict[label[self.labels_order[0]] + '|' + label[self.labels_order[1]]]
    
    def print_table(self, table_data):
        table = Table(show_header=True, header_style="bold black", box=box.HEAVY_EDGE, show_lines=True, show_edge=True)
        for col_name in table_data[0]:
            table.add_column(col_name, justify="center")

        for row in table_data[1:]:
            table.add_row(*row)

        console.print(table)
    
    def generate_labels(self, text_true, text_pred):
        y_true = []
        y_pred = []

        for tt, tp in tqdm(zip(text_true, text_pred), desc="finding labels", total=len(text_true)):
            tp = self.clean_text(tp)

            for wordt, wordp in zip(tt.split(), tp.split()):
                y_true.append(copy.deepcopy(self.get_word_label(wordt)))
                y_pred.append(copy.deepcopy(self.get_word_label(wordp)))

        y_true = torch.tensor(y_true).view(-1)
        y_pred = torch.tensor(y_pred).view(-1)
        
        labels = {
            'All': {
                'y_true': y_true,
                'y_pred': y_pred,
            },
        }
        
        for class_type, class_labels in self.indv_label_map.items():
            target = y_true*0
            preds = y_pred*0

            for label, new_idx in class_labels.items():
                if new_idx != 0:
                    for old_idx in self.label_map[class_type][label]:
                        target[y_true==old_idx] += new_idx
                        preds[y_pred==old_idx] += new_idx
                        
            labels[class_type] = {
                'y_true': target,
                'y_pred': preds,
            }
        
        return labels
    
    def dict2table(self, scores):
        table = [['Label', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'F2-Score', 'F0.5-Score']]
        for key, val in scores.items():
            row = [key]
            for col in table[0][1:]:
                row.append(f"{val[col].item()*100:.2f}")
            table.append(row)
        return table
    
    def calculate_scores(self, y_true, y_pred, print_score=True, returnType='table'):
        labels = self.generate_labels(y_true, y_pred)
        
        scores = {}
        for class_type in self.val_metrics.keys():
            for cat in self.val_metrics[class_type].keys():
                if cat == 'macro_average':
                    name = f'{self.header_map[class_type]} (macro average)'
                    scores[name] = self.val_metrics[class_type][cat]['stats'](labels[class_type]['y_pred'], labels[class_type]['y_true'])
                    scores[name]['Accuracy'] = self.val_metrics[class_type][cat]['Accuracy'](labels[class_type]['y_pred'], labels[class_type]['y_true'])
                else:
                    stats = self.val_metrics[class_type][cat]['stats'](labels[class_type]['y_pred'], labels[class_type]['y_true'])
                    acc = self.val_metrics[class_type][cat]['Accuracy'](labels[class_type]['y_pred'], labels[class_type]['y_true'])
                    
                    for label_type, label_idx in self.indv_label_map[class_type].items():
                        name = f'{self.header_map[class_type]} ({label_type})'
                        scores[name] = {'Accuracy': acc[label_idx]}
                        for k, v in stats.items():
                            scores[name][k] = v[label_idx]
                            
        if returnType=='table':
            scores = self.dict2table(scores)
            
        if print_score:
            if type(scores) == dict:
                table = self.dict2table(scores)
                self.print_table(table)
            else:            
                self.print_table(scores)
            
        return scores