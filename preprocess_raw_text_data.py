"""
Usage: python preprocess_raw_text_data.py --config="example_configs/preprocess_config_en.yaml"
"""

from rich.console import Console
console = Console()

import ray
from ray import serve

console.rule(f"[red] [Initialising Ray]", style="red")

ray.init()
serve.start(detached=False)

import os
from omegaconf import DictConfig, OmegaConf

from transformer_punct_and_capit.dataset.utils.split_text_corpus import split_text_data
from transformer_punct_and_capit.dataset.utils.balance_text_corpus import balance_text_data
from transformer_punct_and_capit.dataset.utils.normalize_text_corpus import NormalizationModel, normalize_text_data

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess Raw Text Data')
    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args()
    return args


def main(cfg):
    NormalizationModel.deploy(lang=cfg.preprocess_config.lang_code)
    norm_handle = NormalizationModel.get_handle()
    
    console.rule(f"[red] [Ray Initialised]", style="red")
    console.print("")

    for dn, fn in cfg.raw_text_corpus.items():
        console.rule(f"[red] [Processing {dn.upper()}]", style="red")

        data_dir = os.path.join(cfg.save_dir, f"{dn}")
        os.makedirs(data_dir, exist_ok=True)
        
        if cfg.preprocess_config.balancing_margin_ratio != -1:
            balance_op_fn = os.path.join(data_dir, f"balanced.txt")
            balance_text_data(fn, balance_op_fn, 
                              punct_labels=cfg.preprocess_config.punct_labels,
                              balancing_margin_ratio=cfg.preprocess_config.balancing_margin_ratio)
        else:
            balance_op_fn = fn
        
        console.rule(f"", style="red")

        normalized_op_fn = os.path.join(data_dir, f"balanced_normalized.txt")
        normalize_text_data(norm_handle, balance_op_fn, normalized_op_fn, punct_labels=cfg.preprocess_config.punct_labels)
        
        console.rule(f"", style="red")

        split_text_data(normalized_op_fn, data_dir, split_ratio=cfg.preprocess_config.split_ratio)

        console.rule(f"[red] [{dn.upper()} processed]", style="red")
        console.print("")
        
if __name__ == '__main__':
    args = parse_arguments()
    cfg = OmegaConf.load(args.config)
    main(cfg)