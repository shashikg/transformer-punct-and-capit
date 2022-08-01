import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import copy
import tarfile
import tempfile
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import torchmetrics
from torchcrf import CRF
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, AutoModel

from ..dataset import Dataset, InferDataset
from ..metric import ClassificationStatScores
from .utils import LinearLayer, ClassificationLayer, applyPunct

from rich import box
from rich.table import Table
from rich.console import Console
                   
class TransformerPunctAndCapitModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
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
        
        # Get pretrained transformer model
        self.transformer = AutoModel.from_pretrained(self.pretrained_model_name)
        if self.freeze_pretrained_model:
            for p in self.transformer.parameters():
                p.requires_grad = False
        
        # Classification layers
        hidden_size = self.transformer.config.hidden_size
        if self.use_bilstm:
            self.bilstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, bidirectional=True)
            self.classifier = ClassificationLayer(in_features=hidden_size*2, out_features=self.num_labels, **cfg.model.classification_head)
        else:
            self.classifier = ClassificationLayer(in_features=hidden_size, out_features=self.num_labels, **cfg.model.classification_head)
        
        # Loss
        if self.use_crf:
            self.crf_loss = CRF(self.num_labels, batch_first=True)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        # Tokenizers and metrics
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer.path)
        [START_ID, PAD_ID, END_ID, UNK_ID] = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token,
                                                                                   self.tokenizer.pad_token,
                                                                                   self.tokenizer.sep_token,
                                                                                   self.tokenizer.unk_token])
        self.token_style = {'START_SEQ': START_ID,
                            'PAD': PAD_ID,
                            'END_SEQ': END_ID,
                            'UNK': UNK_ID}
        
        self.apply_punct = applyPunct(self.labels_dict, self.labels_order)
        
        self.val_score_history = []
        self.setup_metrics()
        
    def setup_metrics(self):
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_labels, average='macro')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_labels, average='macro')
        
        self.train_f1_score = torchmetrics.F1Score(num_classes=self.num_labels, average='macro')
        self.val_f1_score = torchmetrics.F1Score(num_classes=self.num_labels, average='macro')
        
        self.val_classification_score = ClassificationStatScores(num_classes=self.num_labels, average='macro')
        
        self.indv_label_map = {
            'c': {k: v for v, k in enumerate(self.capit_labels)},
            'p': {k: v for v, k in enumerate(self.punct_labels)}
        }
        
        self.label_map = {'c': {}, 'p': {}}
        for key, value in self.labels_dict.items():
            for i in range(2):
                try:
                    self.label_map[self.labels_order[i]][key.split("|")[i]].append(value)
                except:
                    self.label_map[self.labels_order[i]][key.split("|")[i]] = [value]
                    
        self.val_metrics = {}
        for _ in self.labels_order:
            self.val_metrics[_] = {
                'indv': ClassificationStatScores(num_classes=len(self.label_map[_]), average='none'),
                'indv_acc': torchmetrics.Accuracy(num_classes=len(self.label_map[_]), average='none'),
                'avg': ClassificationStatScores(num_classes=len(self.label_map[_]), average='macro'),
                'avg_acc': torchmetrics.Accuracy(num_classes=len(self.label_map[_]), average='macro'),
            }

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
        
    def change_tokenizer(self, new_tokenizer_path):
        self.cfg.tokenizer.path = new_tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer.path)
        
    def save_model(self, filename):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, 'w', encoding='utf-8') as fout:
                OmegaConf.save(config=self.cfg, f=fout, resolve=True)
                
            tokenizer_path = os.path.join(tmpdir, "tokenizer")
            os.makedirs(tokenizer_path, exist_ok=True)
            self.tokenizer.save_pretrained(tokenizer_path)
            
            weights_path = os.path.join(tmpdir, "weights.pt")
            torch.save(self.state_dict(), weights_path)
                
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with tarfile.open(filename, "w:") as tar:
                tar.add(tmpdir, arcname=".")
    
    @classmethod
    def restore_model(cls, filename, device='cuda'):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} does not exist")
            
        with tempfile.TemporaryDirectory() as tmpdir:
            tar = tarfile.open(filename, "r:")
            tar.extractall(path=tmpdir)
            tar.close()
            
            config_path = os.path.join(tmpdir, "config.yaml")
            cfg = OmegaConf.load(config_path)
            cfg.tokenizer.path = str(os.path.join(tmpdir, "tokenizer"))
            
            instance = cls(cfg)
            weights_path = os.path.join(tmpdir, "weights.pt")
            instance.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
        
        instance = instance.to(device)
        return instance
        
    def forward(self, x, attn_masks):
        x = self.transformer(x, attention_mask=attn_masks)[0]
        
        if self.use_bilstm:
            x = torch.transpose(x, 0, 1) # (B, N, E) -> (N, B, E)
            x, (_, _) = self.bilstm(x)
            x = torch.transpose(x, 0, 1) # (N, B, E) -> (B, N, E)
            
        x = self.classifier(x)
        return x
    
    def get_loss(self, x, attn_masks, y):
        x = self.forward(x, attn_masks)
        
        if self.use_crf:
            attn_masks = attn_masks.byte()
            loss = -self.crf_loss(x, y, mask=attn_masks, reduction='token_mean')
            
            with torch.no_grad():
                dec_out = self.crf_loss.decode(x, mask=attn_masks)
                y_pred = torch.zeros(y.shape).long().to(y.device)
                for i in range(len(dec_out)):
                    y_pred[i, :len(dec_out[i])] = torch.tensor(dec_out[i]).to(y.device)
        else:
            y_pred = x.view(-1, x.shape[2])
            y = y.view(-1)
            loss = self.ce_loss(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1).view(-1)
            
        return loss, y_pred
    
    def get_pred(self, x, attn_mask):
        x, attn_mask = x.to(self.device), attn_mask.to(self.device)
        y = torch.zeros(x.shape)
        y = y.to(self.device)
        
        x = self.forward(x, attn_mask)
        
        if self.use_crf:
            attn_mask = attn_mask.byte()
            dec_out = self.crf_loss.decode(x, mask=attn_mask)
            y_pred = torch.zeros(y.shape).long().to(y.device)
            for i in range(len(dec_out)):
                y_pred[i, :len(dec_out[i])] = torch.tensor(dec_out[i]).to(y.device)
        else:
            y_pred = torch.argmax(x, dim=-1)
            
        return y_pred
    
    def training_step(self, batch, batch_idx):
        x, y, attn_masks, y_mask = batch
        loss, y_pred = self.get_loss(x, attn_masks, y)
        
        with torch.no_grad():
            y_mask = y_mask.view(-1)
            y = y.view(-1)
            y_pred = y_pred.view(-1)
            y_true = y[y_mask==1]
            y_pred = y_pred[y_mask==1]
            self.train_acc(y_pred, y_true)
            self.train_f1_score(y_pred, y_true)
            
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        self.log("train_f1", self.train_f1_score, on_step=True, on_epoch=False, logger=True, prog_bar=False)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, attn_masks, y_mask = batch
        loss, y_pred = self.get_loss(x, attn_masks, y)
        
        y_mask = y_mask.view(-1)
        y = y.view(-1)
        y_pred = y_pred.view(-1)
        
        y_true = y[y_mask==1]
        y_pred = y_pred[y_mask==1]
        
        self.val_acc.update(y_pred, y_true)
        self.val_f1_score.update(y_pred, y_true)
        self.val_classification_score.update(y_pred, y_true)
        
        self.log("val_loss", loss, logger=True, prog_bar=True)

        for class_type, class_labels in self.indv_label_map.items():
            target = y_true*0
            preds = y_pred*0
            
            for label_type, label_idx in class_labels.items():
                if label_idx != 0:
                    for old_idx in self.label_map[class_type][label_type]:
                        target[y_true==old_idx] += label_idx
                        preds[y_pred==old_idx] += label_idx
        
            for _, metric_fn in self.val_metrics[class_type].items():
                metric_fn.update(preds.to('cpu'), target.to('cpu'))
        
    def validation_epoch_end(self, outputs):
        val_acc = self.val_acc.compute().item()
        val_f1_score = self.val_f1_score.compute().item()
        self.val_acc.reset()
        self.val_f1_score.reset()
        
        val_classification_score = self.val_classification_score.compute()
        self.val_classification_score.reset()
        
        self.log('val_acc', val_acc, logger=True, prog_bar=True)
        self.log('val_f1', val_f1_score, logger=True, prog_bar=False)
        
        try:
            if val_f1_score > min(self.val_score_history):
                self.save_model(f'{self.cfg.exp_dir}/pcm_checkpoints/epoch_{len(self.val_score_history)+1}-val_{int(val_f1_score*100)}.pcm')
        except:
            self.save_model(f'{self.cfg.exp_dir}/pcm_checkpoints/epoch_{len(self.val_score_history)+1}-val_{int(val_f1_score*100)}.pcm')
                
        self.val_score_history.append(val_f1_score)
        self.print_validation_summary(val_acc, val_classification_score)
        
    def print_validation_summary(self, val_acc, val_classification_score):
        classification_report = {}
        for class_type in self.labels_order:
            classification_report[class_type] = {}
            for metric_type, metric_fn in self.val_metrics[class_type].items():
                classification_report[class_type][metric_type] = metric_fn.compute()
                metric_fn.reset()
                
        classification_table = [["Name", "Accuracy", "Precision", "Recall", "F1-Score", 'F2-Score', 'F0.5-Score']]
        pre_text_name = {'p': "Punct ", 'c': "Capit "}
        
        for class_type in self.labels_order:
            for metric_type, matric_score in classification_report[class_type].items():
                if "acc" not in metric_type:
                    if 'avg' in metric_type:
                        row = [f"[bold]{pre_text_name[class_type]}(macro avg)[/bold]", 
                               str(round(classification_report[class_type][f"{metric_type}_acc"].item()*100, 2))]
                        
                        for key in ['Precision', 'Recall', 'F1-Score', 'F2-Score', 'F0.5-Score']:
                            row.append(str(round(classification_report[class_type][f"{metric_type}"][key].item()*100, 2)))
                            
                        classification_table.append(row)
                    else:
                        for label_type, label_idx in self.indv_label_map[class_type].items():
                            print()
                            row = [f"[bold]{pre_text_name[class_type]}({label_type})[/bold]", 
                                   str(round(classification_report[class_type][f"{metric_type}_acc"][label_idx].item()*100, 2))]
                        
                            for key in ['Precision', 'Recall', 'F1-Score', 'F2-Score', 'F0.5-Score']:
                                row.append(str(round(classification_report[class_type][f"{metric_type}"][key][label_idx].item()*100, 2)))
                    
                            classification_table.append(row)
                        
            classification_table.append(5*["-----"])
            
        row = ["[bold]All (macro avg)[/bold]", 
               str(round(val_acc*100, 2))]
        for key in ['Precision', 'Recall', 'F1-Score', 'F2-Score', 'F0.5-Score']:
            row.append(str(round(val_classification_score[key].item()*100, 2)))
        classification_table.append(row)

        console = Console()
        table = Table(show_header=True, header_style="bold black", expand=False, box=box.HEAVY_EDGE, show_lines=True, show_edge=True)
        for col_name in classification_table[0]:
            table.add_column(col_name, justify="center")
        
        for row in classification_table[1:]:
            table.add_row(*row)
            
        console.print(table)
        console.print("\n")
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.cfg.optim.lr, weight_decay=self.cfg.optim.weight_decay)
    
    def train_dataloader(self):
        data_set = Dataset(data_path=self.cfg.dataset.train_ds.parsed_data_file,
                            tokenizer=self.tokenizer, sequence_len=self.cfg.dataset.max_seq_length,
                            augmentation_args=self.cfg.dataset.augmentation,
                            is_train=True)
        
        data_loader_params = {
            'batch_size': self.cfg.dataset.train_ds.batch_size,
            'shuffle': self.cfg.dataset.train_ds.shuffle,
            'num_workers': self.cfg.dataset.train_ds.num_workers
        }
        
        return DataLoader(data_set, **data_loader_params)

    def val_dataloader(self):
        data_set = Dataset(data_path=self.cfg.dataset.dev_ds.parsed_data_file,
                            tokenizer=self.tokenizer, sequence_len=self.cfg.dataset.max_seq_length,
                            is_train=False)
        
        data_loader_params = {
            'batch_size': self.cfg.dataset.dev_ds.batch_size,
            'shuffle': self.cfg.dataset.dev_ds.shuffle,
            'num_workers': self.cfg.dataset.dev_ds.num_workers
        }
        
        return DataLoader(data_set, **data_loader_params)
    
    def input_example(self, max_batch=1, max_dim=64):
        sample = next(self.parameters())
        sz = (max_batch, max_dim)
        x = torch.randint(low=0, high=self.tokenizer.vocab_size, size=sz, device=sample.device)
        attn_masks = torch.randint(low=0, high=2, size=sz, device=sample.device)
        
        input_dict = {
            "x": x,
            "attn_masks": attn_masks,
        }
        
        return input_dict
    
    def export_model(self, export_path, quantize=True, device='cpu'):
        if len(export_path.split('.')) > 1:
            export_path = f"{'.'.join(export_path.split('.')[:-1])}.epcm"
        else:
            export_path = export_path + ".epcm"

        model = self.to('cpu')
        model = model.eval()

        if quantize:
            quant_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            quant_model = model
            
        example = model.input_example()
        traced_model = torch.jit.trace(quant_model, (example['x'], example['attn_masks']))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, 'w', encoding='utf-8') as fout:
                OmegaConf.save(config=self.cfg, f=fout, resolve=True)
                
            tokenizer_path = os.path.join(tmpdir, "tokenizer")
            os.makedirs(tokenizer_path, exist_ok=True)
            self.tokenizer.save_pretrained(tokenizer_path)
            
            if self.use_crf:
                crf_layer_path = os.path.join(tmpdir, "crf_layer.pt")
                torch.save(self.crf_loss.state_dict(), crf_layer_path)
            
            jit_model_path = os.path.join(tmpdir, "jit_model.pt")
            torch.jit.save(traced_model, jit_model_path)
            
            if len(os.path.dirname(export_path)):
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            with tarfile.open(export_path, "w:") as tar:
                tar.add(tmpdir, arcname=".")
                
        print(f"[export_model]: Model exported to {export_path}")
        return export_path
    
    def infer_dataloader(self, text_list, batch_size=128, max_sequence_length=64, show_pbar=True):
        data_items = []
        
        if show_pbar:
            data_iter = tqdm(range(len(text_list)), desc="tokenizing data")
        else:
            data_iter = range(len(text_list))
        
        for idx in data_iter:
            text = text_list[idx]
            words = text.split()
            word_pos = 0

            while word_pos < len(words):
                x = [self.token_style['START_SEQ']]
                y_mask = [1]

                while (len(x) < max_sequence_length) and (word_pos < len(words)):
                    tokens = self.tokenizer.tokenize(words[word_pos])
                    if len(tokens) + len(x) >= max_sequence_length:
                        break
                    else:
                        for i in range(len(tokens)-1):
                            x.append(self.tokenizer.convert_tokens_to_ids(tokens[i]))
                            y_mask.append(0)
                        
                        if len(tokens) > 0:
                            x.append(self.tokenizer.convert_tokens_to_ids(tokens[-1]))
                        else:
                            x.append(self.token_style['UNK'])
                            
                        y_mask.append(1)
                        word_pos += 1

                x.append(self.token_style['END_SEQ'])
                y_mask.append(1)

                if len(x) < max_sequence_length:
                    x = x + [self.token_style['PAD'] for _ in range(max_sequence_length - len(x))]
                    y_mask = y_mask + [0 for _ in range(max_sequence_length - len(y_mask))]

                attn_mask = [1 if token != self.token_style['PAD'] else 0 for token in x]
                data_items.append(copy.deepcopy([idx, x, attn_mask, y_mask]))
                
        data_loader_params = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 8
        }
        
        data_set = InferDataset(data_items)
        data_loader = torch.utils.data.DataLoader(data_set, **data_loader_params)
        
        return data_loader
    
    @torch.no_grad()
    def predict_batch(self, text_list, batch_size=64, max_sequence_length=64, show_pbar=True):
        mode = self.training
        self.eval()
        
        data_loader = self.infer_dataloader(text_list, batch_size, max_sequence_length, show_pbar=show_pbar)
        
        predictions = []
        prev_sent_id = -1
        result = ""
        decode_idx = 0
        words = []
        
        with torch.no_grad():
            if show_pbar:
                data_iter = tqdm(data_loader, desc='predicting')
            else:
                data_iter = data_loader
                
            for sent_idx, x, attn_mask, y_mask in data_iter:
                y_predict = self.get_pred(x, attn_mask)
                
                for n, sent_id in enumerate(sent_idx):
                    if prev_sent_id != sent_id:
                        if prev_sent_id >= 0:
                            predictions.append(copy.deepcopy(result.strip()))
                            
                        result = ""
                        decode_idx = 0
                        words = text_list[sent_id].split()
                        
                    for i in range(y_mask[n].shape[0]):
                        if x[n][i] == self.token_style['START_SEQ']:
                            continue
                            
                        if x[n][i] == self.token_style['END_SEQ']:
                            break
                            
                        if y_mask[n][i] == 1:
                            try:
                                result += self.apply_punct(words[decode_idx], y_predict[n][i].item()) + ' '
                            except:
                                pass

                            decode_idx += 1
                                
                    prev_sent_id = sent_id
                    
            predictions.append(copy.deepcopy(result.strip()))
        
        self.train(mode=mode)
        
        return predictions
        
    def predict(self, text, max_sequence_length=64):
        return self.predict_batch([text], batch_size=64, max_sequence_length=max_sequence_length, show_pbar=False)
        
        
        
        
        
        

                                 

