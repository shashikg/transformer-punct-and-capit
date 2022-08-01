# Tranformer Based Model for Punctuation and Capitalization Restoration

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
