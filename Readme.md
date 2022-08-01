# BERT Based Model for Punctuation and Capitalization Restoration

## Features:

* Uses Huggingface Tranformer library for base transformer architecture. 
* Pytorch Lightning is used for training and checkpoints.
* Easy config based model description for easy experimenttation and reaearch.
* Can be exported as a pytorch quantized model for faster inference on CPU.
* Includes helper function for data preparation, text normalization, and offline sentence augmentation specific for punctuation and capitalization restoration. 

## Quick guide:

```sh

# Install requirements:
pip install -r requirements.txt

# Downloads raw text corpus from tatoeba for english language
bash download_tatoeba_en_sent.sh

# Preprocess raw text data. Check config file for more details
python preprocess_raw_text_data.py --config="example_configs/preprocess_config_en.yaml"

# Merge multiple data files into one, apply sent augmentation, and tokenization. Check config file for more details
python merge_and_tokenize_datasets.py --config="example_configs/model_config_en.yaml"

# Merge multiple data files into one, apply sent augmentation, and tokenization. Check config file for more details
python train_punct_and_capit_model.py --config="example_configs/model_config_en.yaml"

```
