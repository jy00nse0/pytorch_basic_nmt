#!/bin/sh
git clone https://github.com/jy00nse0/pytorch_basic_nmt.git
cd pytorch_basic_nmt
conda env create -f environment.yml
cd data
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/17bmdohiXGQDd6DE1pDl0VvNV702K8uuG/view?usp=sharing
tar -xvf wmt14_tok_len50.tar
cd ../
conda activate pytorch_nmt
chmod +x scripts/train.sh

#tmux new-session -d -s nmt_train \
#"bash scripts/train.sh 2>&1 | tee train_$(date +%Y%m%d_%H%M%S).log"
