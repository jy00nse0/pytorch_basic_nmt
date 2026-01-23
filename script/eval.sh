#!/bin/sh
# chmod +x scripts/eval.sh
# bash ./scripts/eval.sh 2>&1 | tee eval_$(date +%Y%m%d_%H%M%S).log

vocab="data/vocab.json"
train_src="data/wmt14_tok_len50/train.en"
train_tgt="data/wmt14_tok_len50/train.de"
dev_src="data/wmt14_tok_len50/validation.en"
dev_tgt="data/wmt14_tok_len50/validation.de"
test_src="data/wmt14_tok_len50/test.en"
test_tgt="data/wmt14_tok_len50/test.de"

work_dir="/workspace/T1_base_nmt"

# decoding
python nmt.py \
    decode \
    --cuda \
    --beam-size 1 \
    --max-decoding-time-step 50 \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt


perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt
