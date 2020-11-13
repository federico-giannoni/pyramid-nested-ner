#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -e

echo "downloading genia splits..."

mkdir -p "$DIR/data/iob2"
git clone https://github.com/thecharm/boundary-aware-nested-ner
cp boundary-aware-nested-ner/Our_boundary-aware_model/data/genia/* "$DIR/data/iob2"
rm -rf boundary-aware-nested-ner
mkdir -p "$DIR/data/rasa"

python "$DIR/iob2_to_rasa.py" "$DIR/data/iob2" "$DIR/data/rasa"

echo "downloading PubMed biovec word embeddings..."

gdown https://drive.google.com/u/0/uc?id=0BzMCqpcgEJgiUWs0ZnU0NlFTam8&export=download
until [[ -f "bio_nlp_vec.tar.gz" ]]               # gdown is async...
do
    sleep 1
done
sleep 5
tar -xzf bio_nlp_vec.tar.gz && mv bio_nlp_vec "$DIR/data/bio_nlp_vec" && rm bio_nlp_vec.tar.gz
curl https://raw.githubusercontent.com/spyysalo/wvlib/master/wvlib.py --output "$DIR/wvlib.py"

echo "downlading bioBERT..."

gdown https://drive.google.com/u/0/uc?id=1GJpGjQj6aZPV-EfbiQELpBkvlGtoKiyA&export=download
until [[ -f "biobert_large_v1.1_pubmed.tar.gz" ]]  # gdown is async...
do
    sleep 1
done
sleep 5
tar -xzf biobert_large_v1.1_pubmed.tar.gz && mv biobert_large "$DIR/biobert_large" && rm biobert_large_v1.1_pubmed.tar.gz
mv "$DIR/biobert_large/*vocab*.txt" "$DIR/biobert_large/vocab.txt"
mv "$DIR/biobert_large/*config*.json" "$DIR/biobert_large/config.json"
mv "$DIR/biobert_large/*ckpt.meta" "$DIR/biobert_large/model.ckpt.meta"
mv "$DIR/biobert_large/*ckpt.index" "$DIR/biobert_large/model.ckpt.index"
mv "$DIR/biobert_large/bio_bert_large_1000k.ckpt.data-00000-of-00001" "$DIR/biobert_large/model.ckpt.data-00000-of-00001"
