#!/bin/sh
# chmod +x scripts/eval.sh
# bash ./scripts/eval.sh 2>&1 | tee eval_$(date +%Y%m%d_%H%M%S).log

# replace --att-type,--att-score location args with model setting

vocab="data/vocab.json"
train_src="data/wmt14_tok_len50/train.en"
train_tgt="data/wmt14_tok_len50/train.de"
dev_src="data/wmt14_tok_len50/test.en"
dev_tgt="data/wmt14_tok_len50/test.de"
test_src="data/wmt14_tok_len50/test.en"
test_tgt="data/wmt14_tok_len50/test.de"

work_dir="/workspace/T1_base_reverse_nmt_global_location_feedinput"

mkdir -p ${work_dir}
echo save results to ${work_dir}
python nmt.py \
    decode \
    --cuda \
    --beam-size 1 \
    --max-decoding-time-step 50 \
    --att-type global \
    --att-score location \
    --window-size 10 \
    --reverse \
    --unk-replace \
    --unk-dict data/dict.en-de.json \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt



perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt
