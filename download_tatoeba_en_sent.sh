mkdir -p data/raw_data
cd data/raw_data

echo "Downlaod data.."
wget https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2

echo "Extracting data.."
bzip2 -dk eng_sentences.tsv.bz2

echo "Cleanup.."
rm eng_sentences.tsv.bz2