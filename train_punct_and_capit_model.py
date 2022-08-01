"""
Usage: python train_punct_and_capit_model.py --config="example_configs/model_config_en.yaml"
"""

from transformer_punct_and_capit.models import TransformerPunctAndCapitModel
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--model_name', default="none", type=str, help='config file')
    args = parser.parse_args()
    return args

def main(cfg):
    if cfg.experiment_details.model_name == None:
        cfg.experiment_details.model_name = cfg.model.pretrained_model_name.split("/")[-1]
        if cfg.model.use_crf:
            cfg.experiment_details.model_name += "-crf"
            
        if cfg.model.use_bilstm:
            cfg.experiment_details.model_name += "-bilstm"
            
        cfg.experiment_details.model_name += f"-lr_{cfg.optim.lr}-epoch_{cfg.pl_trainer.epochs}"
        
    cfg.exp_dir = f"{cfg.experiment_details.save_dir}/{cfg.experiment_details.model_name}"
            
    model = TransformerPunctAndCapitModel(cfg)

    trainer = pl.Trainer(gpus=[0], 
                         accelerator="gpu",
                         default_root_dir=cfg.exp_dir, **cfg.pl_trainer)

    trainer.fit(model)


if __name__ == '__main__':
    args = parse_arguments()
    cfg = OmegaConf.load(args.config)

    if args.model_name != "none":
        cfg.experiment_details.model_name = args.model_name
        
    main(cfg)