#tmux new-session -d -s nmt_train \
#"bash scripts/train.sh 2>&1 | tee train_$(date +%Y%m%d_%H%M%S).log"

#tmux attach -t nmt_train
#tmux ls
#tmux kill-session -t nmt_train


#!/bin/sh

vocab="data/vocab.json"
train_src="data/wmt14_tok_len50/train.en"
train_tgt="data/wmt14_tok_len50/train.de"
dev_src="data/wmt14_tok_len50/validation.en"
dev_tgt="data/wmt14_tok_len50/validation.de"
test_src="data/wmt14_tok_len50/test.en"
test_tgt="data/wmt14_tok_len50/test.de"

work_dir="/workspace/T1_base_nmt"

mkdir -p ${work_dir}
echo save results to ${work_dir}

# training
python nmt.py \
    train \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --input-feed \
    --valid-niter 8081 \
    --batch-size 128 \
    --hidden-size 1000 \
    --embed-size 1000 \
    --uniform-init 0.1 \
    --label-smoothing 0.0 \
    --dropout 0.0 \
    --clip-grad 5.0 \
    --max-epoch 10 \
    --lr 1.0 \
    --save-to ${work_dir}/model.bin \
    --lr-decay 0.5 \
    --no-attention \
    --lr-decay-start 5
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
