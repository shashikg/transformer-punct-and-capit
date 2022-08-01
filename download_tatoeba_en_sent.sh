mkdir -p data/raw_data
cd data/raw_data

echo "Downlaod data.."
wget https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2

echo "Extracting data.."
bzip2 -dk eng_sentences.tsv.bz2

cd ../..
echo "Extract sentences from tsv file.."
python tatoeba_prepare.py --tsv_fn='data/raw_data/eng_sentences.tsv' --op_txt_fn='data/raw_data/tatoeba.txt'

cd data/raw_data
echo "Cleanup.."
rm eng_sentences.tsv.bz2
rm eng_sentences.tsv