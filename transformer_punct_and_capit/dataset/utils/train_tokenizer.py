import os
import json
from transformers import AutoTokenizer

def train_tokenizer(pretrained_model_name, save_dir, data_file=None, batch_size=128):
    save_path = f"{save_dir}/tokenizers/{pretrained_model_name.split('/')[-1]}"
    os.makedirs(save_path, exist_ok=True)
    
    print(f"[train_tokenizer]: Loading pretrained tokenizer..")
    pt_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    if data_file:
        print(f"[train_tokenizer]: Loading data")

        dataset = []
        with open(data_file, 'r') as fio:
            for line in fio:
                sample = json.loads(line)
                dataset.append(sample["x_text"])

        def batch_iterator(batch_size=batch_size):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i:i+batch_size]

        print(f"[train_tokenizer]: Dataset loaded!")

        print(f"[train_tokenizer]: Training new tokenizer..")
        new_tokenizer = pt_tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=pt_tokenizer.vocab_size)

        new_tokenizer.save_pretrained(save_path)
        print(f"[train_tokenizer]: New tokenizer saved at: {save_path}")
    else:
        print(f"[train_tokenizer]: Empty data_file saving pre-trained tokenizer at: {save_path}")
        pt_tokenizer.save_pretrained(save_path)
    
    return save_path
