#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -e

echo "downloading genia splits..."

mkdir -p "$DIR/data/iob2"
git clone https://github.com/thecharm/boundary-aware-nested-ner
cp boundary-aware-nested-ner/Our_boundary-aware_model/data/genia/* "$DIR/data/iob2"
rm -rf boundary-aware-nested-ner
mkdir -p "$DIR/data/rasa"

python "$DIR/utils/iob2_to_rasa.py" "$DIR/data/iob2" "$DIR/data/rasa"

python "$DIR/utils/download.py"
# gdown https://drive.google.com/u/0/uc?id=0BzMCqpcgEJgiUWs0ZnU0NlFTam8&export=download
# until [[ -f "bio_nlp_vec.tar.gz" ]]               # gdown is async...
# do
#    sleep 1
# done
# gdown https://drive.google.com/u/0/uc?id=1GJpGjQj6aZPV-EfbiQELpBkvlGtoKiyA&export=download
# until [[ -f "biobert_large_v1.1_pubmed.tar.gz" ]]  # gdown is async...
# do
#    sleep 1
# done
# sleep 3
[[ ! -f "bio_nlp_vec.tar.gz" ]] || echo "bio_nlp_vec.tar.gz was not downloaded." && exit 1
[[ ! -f "biobert_large_v1.1_pubmed.tar.gz" ]] || echo "biobert_large_v1.1_pubmed.tar.gz was not downloaded." && exit 1

tar -xzf bio_nlp_vec.tar.gz && mv bio_nlp_vec "$DIR/data/" && rm bio_nlp_vec.tar.gz
tar -xzf biobert_large_v1.1_pubmed.tar.gz && mv biobert_large "$DIR/data/" && rm biobert_large_v1.1_pubmed.tar.gz
# curl https://raw.githubusercontent.com/spyysalo/wvlib/master/wvlib.py --output "$DIR/wvlib.py"
mv "$DIR/data/biobert_large/vocab_cased_pubmed_pmc_30k.txt" "$DIR/data/biobert_large/vocab.txt"
mv "$DIR/data/biobert_large/bert_config_bio_58k_large.json" "$DIR/data/biobert_large/config.json"
mv "$DIR/data/biobert_large/bio_bert_large_1000k.ckpt.meta" "$DIR/data/biobert_large/model.ckpt.meta"
mv "$DIR/data/biobert_large/bio_bert_large_1000k.ckpt.index" "$DIR/data/biobert_large/model.ckpt.index"
mv "$DIR/data/biobert_large/bio_bert_large_1000k.ckpt.data-00000-of-00001" "$DIR/data/biobert_large/model.ckpt.data-00000-of-00001"
