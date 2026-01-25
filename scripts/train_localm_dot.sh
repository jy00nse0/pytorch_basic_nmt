# T1- Base + reverse + dropout + local-p attention (general) + feed input

#chmod +x scripts/train_localm_dot.sh
#tmux new-session -d -s nmt_train \
#"bash scripts/train_localm_dot.sh 2>&1 | tee train_T4_localm_dot$(date +%Y%m%d_%H%M%S).log"

#tmux attach -t nmt_train
#tmux ls
#tmux kill-session -t nmt_train

#!/bin/sh

vocab="data/vocab.json"
train_src="data/wmt14_tok_len50/train.en"
train_tgt="data/wmt14_tok_len50/train.de"
dev_src="data/wmt14_tok_len50/test.en"
dev_tgt="data/wmt14_tok_len50/test.de"
test_src="data/wmt14_tok_len50/test.en"
test_tgt="data/wmt14_tok_len50/test.de"

work_dir="/workspace/T4_localm_dot"

mkdir -p ${work_dir}
echo save results to ${work_dir}
# T1_base_reverse
# training
python nmt.py \
    train \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${test_src} \
    --dev-tgt ${test_tgt} \
    --input-feed \
    --valid-niter 4000 \
    --batch-size 128 \
    --hidden-size 1000 \
    --embed-size 1000 \
    --uniform-init 0.1 \
    --label-smoothing 0.0 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --max-epoch 12 \
    --lr 1.0 \
    --save-to ${work_dir}/model.bin \
    --lr-decay 0.5 \
    --att-type local-m \
    --att-score dot \
    --lr-decay-start 8 \
    --reverse 
# decoding
python nmt.py \
    decode \
    --cuda \
    --beam-size 1 \
    --max-decoding-time-step 50 \
    --att-type local-m \
    --att-score dot \
    --window-size 10 \
    --reverse \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt


perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt
