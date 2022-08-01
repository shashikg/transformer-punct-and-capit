"""
Usage: python merge_and_tokenize_datasets.py --config="example_configs/model_config_en.yaml"
"""

from rich.console import Console
console = Console()

import os
from omegaconf import DictConfig, OmegaConf

from transformer_punct_and_capit.dataset.utils.create_manifest import create_manifest
from transformer_punct_and_capit.dataset.utils.train_tokenizer import train_tokenizer
from transformer_punct_and_capit.dataset.utils.tokenize_manifest import tokenize_manifest
from transformer_punct_and_capit.dataset.utils import merge_multiple_text_data, sent_augmentation
from transformer_punct_and_capit.dataset.utils import writeTextData, readTextData, generate_labels_dict, readJsonManifest

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='Merge and Tokenize Data')
    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args()
    return args

def main(config_path):
    cfg = OmegaConf.load(config_path)
    cfg = generate_labels_dict(cfg)
    
    for split in ['dev', 'train']:
        console.rule(f"[red] [Merging {split} set]", style="red")
        text_files_to_merge = []
        for dn in cfg.dataset[f"{split}_ds"].datasets.split(","):
            text_files_to_merge.append(f"{cfg.dataset.processed_data_dir}/{dn}/balanced_normalized_{split}.txt")

        merged_corpus = merge_multiple_text_data(text_files_to_merge, 
                                                 max_lines=cfg.dataset[f"{split}_ds"].per_data_max_sentence)

        console.print("")
        console.rule(f"[red] [Applying sent augmentation]", style="red")
        merged_corpus = sent_augmentation.combine_sentences(merged_corpus, 
                                                            combine_ratio=cfg.dataset.sent_augmentation.combine_ratio)

        merged_corpus = sent_augmentation.cut_sentences(merged_corpus, 
                                                        cut_ratio=cfg.dataset.sent_augmentation.cut_ratio)
        
        merged_corpus = list(set(merged_corpus))
        writeTextData(f"{cfg.dataset.data_dir}/{split}.txt", merged_corpus)

        console.print("")
        console.rule(f"[red] [Generating JSON label files]", style="red")
        man_fn = create_manifest(f"{cfg.dataset.data_dir}/{split}.txt",
                                 cfg.dataset.data_dir,
                                 punct_labels=cfg.dataset.punct_labels,
                                 labels_order=cfg.dataset.labels_order)
        
        console.rule(f"", style="red")
        console.print("")
        
    if (cfg.tokenizer.train_new or cfg.tokenizer.path == None):
        console.rule(f"[red] [Training tokenizer]", style="red")
        
        if cfg.tokenizer.use_pretrained_tokenizer:
            cfg.tokenizer.path = train_tokenizer(pretrained_model_name=cfg.tokenizer.pretrained_model_name,
                                                 save_dir=cfg.experiment_details.save_dir)
        else:
            cfg.tokenizer.path = train_tokenizer(pretrained_model_name=cfg.tokenizer.pretrained_model_name,
                                                 save_dir=cfg.experiment_details.save_dir,
                                                 data_file=f"{cfg.dataset.data_dir}/train.json")
            
        console.rule(f"", style="red")
        console.print("")
        
    for split in ['dev', 'train']:
        console.rule(f"[red] [Tokenizing {split} set]", style="red")
        cfg.dataset[f"{split}_ds"].parsed_data_file = tokenize_manifest(man_fn=f"{cfg.dataset.data_dir}/{cfg.dataset[f'{split}_ds'].json_data_file}",
                                                                        tokenizer_path=cfg.tokenizer.path,
                                                                        labels_dict=cfg.model.labels_dict,
                                                                        max_seq_length=cfg.dataset.max_seq_length)
        console.rule(f"", style="red")
        console.print("")
        
    with open(config_path, 'w', encoding='utf-8') as fout:
        OmegaConf.save(config=cfg, f=fout, resolve=True)
        
if __name__ == '__main__':
    args = parse_arguments()
    main(args.config)