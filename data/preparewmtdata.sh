#download
%cd data
gdown --fuzzy https://drive.google.com/file/d/1p3VADtNVs_0BUADXclc_BmA_ZgJPCyd-/view?usp=sharing
tar -xvf wmt14_tok_len50.tar.gz
#make vocab
python vocab.py \
    --train-src=data/wmt14_tok_len50/train.en \
    --train-tgt=data/wmt14_tok_len50/train.de \
    data/vocab.json
