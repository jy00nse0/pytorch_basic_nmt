# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --label-smoothing=<float>               use label smoothing [default: 0.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --no-attention                          disable attention mechanism [default: False]
    --num-layers=<int>                      number of layers [default: 4]
    --reverse                               reverse source sentence for NMT
    --att-type=<str>                        attention type (global, local-m, local-p) [default: global]
    --att-score=<str>                       attention score function (dot, general, location) [default: dot]
    --window-size=<int>                     window size D for local attention [default: 10]
    --lr-decay-start=<int>                 start learning rate decay after this epoch [default: 10]
    --use-all-layer-hiddenstates            use hidden states from all layers [default: False]
    --unk-replace                           replace unknown words
    --unk-dict=<file>                       unknown word dictionary [default: data/dict.en-de.json]
"""

import json
import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from utils import read_corpus, batch_iter, LabelSmoothingLoss
from vocab import Vocab, VocabEntry


Hypothesis = namedtuple('Hypothesis', ['value', 'score', 'attention_history'])


class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, input_feed=True, label_smoothing=0., use_attention=True, num_layers=4, use_all_layer_hiddenstates=False):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.input_feed = input_feed
        self.use_attention = use_attention
        self.num_layers = num_layers
        self.use_all_layer_hiddenstates = use_all_layer_hiddenstates

        # initialize neural network layers...

        self.src_embed = nn.Embedding(len(vocab.src), embed_size, padding_idx=vocab.src['<pad>'])
        self.tgt_embed = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=vocab.tgt['<pad>'])

        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, bidirectional=False, dropout=dropout_rate if num_layers > 1 else 0.)
        
        decoder_lstm_input = embed_size + hidden_size if self.input_feed else embed_size
        
        # Use ModuleList for multi-layer decoder
        self.decoder_lstms = nn.ModuleList()
        for i in range(num_layers):
            input_size = decoder_lstm_input if i == 0 else hidden_size
            self.decoder_lstms.append(nn.LSTMCell(input_size, hidden_size))

        # attention: dot product attention
        # project source encoding to decoder rnn's state space
        if self.use_attention:
            self.att_src_linear = nn.Linear(hidden_size, hidden_size, bias=False)

            # transformation of decoder hidden states and context vectors before reading out target words
            # this produces the `attentional vector` in (Luong et al., 2015)
            self.att_vec_linear = nn.Linear(hidden_size + hidden_size, hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(hidden_size, len(vocab.tgt), bias=False)

        # dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

        # initialize the decoder's state and cells with encoder hidden states
        #self.decoder_cell_init = nn.Linear(hidden_size, hidden_size)
        self.decoder_cell_init = None

        self.label_smoothing = label_smoothing
        if label_smoothing > 0.:
            self.label_smoothing_loss = LabelSmoothingLoss(label_smoothing,
                                                           tgt_vocab_size=len(vocab.tgt), padding_idx=vocab.tgt['<pad>'])

        self.att_type = None
        self.att_score = None
        self.window_size = 10
        self.att_Wa = None
        self.att_Wp = None
        self.att_vp = None

    def set_attention_config(self, att_type='global', att_score='dot', window_size=10):
        self.att_type = att_type
        self.att_score = att_score
        self.window_size = window_size
        
        print(f"Attenuation Config: Type={att_type}, Score={att_score}, WindowSize={window_size}", file=sys.stderr)

        if self.use_attention:
            if att_score == 'general':
                self.att_Wa = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            elif att_score == 'location':
                self.att_Wa = nn.Linear(self.hidden_size, 100, bias=False)
                print("Location Score Size: 100", file=sys.stderr)
            
            if att_type == 'local-p':
                self.att_Wp = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.att_vp = nn.Linear(self.hidden_size, 1, bias=False)

    @property
    def device(self) -> torch.device:
        return self.src_embed.weight.device

    def forward(self, src_sents: List[List[str]], tgt_sents: List[List[str]]) -> torch.Tensor:
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """

        # (src_sent_len, batch_size)
        src_sents_var = self.vocab.src.to_input_tensor(src_sents, device=self.device)
        # (tgt_sent_len, batch_size)
        tgt_sents_var = self.vocab.tgt.to_input_tensor(tgt_sents, device=self.device)
        src_sents_len = [len(s) for s in src_sents]

        src_encodings, decoder_init_vec = self.encode(src_sents_var, src_sents_len)

        src_sent_masks = self.get_attention_mask(src_encodings, src_sents_len)

        src_len_tensor = torch.tensor(src_sents_len, dtype=torch.float, device=self.device)

        # (tgt_sent_len - 1, batch_size, hidden_size)
        att_vecs = self.decode(src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var[:-1], src_len_tensor=src_len_tensor)

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        tgt_words_log_prob = F.log_softmax(self.readout(att_vecs), dim=-1)

        if self.label_smoothing:
            # (tgt_sent_len - 1, batch_size)
            tgt_gold_words_log_prob = self.label_smoothing_loss(tgt_words_log_prob.view(-1, tgt_words_log_prob.size(-1)),
                                                                tgt_sents_var[1:].view(-1)).view(-1, len(tgt_sents))
        else:
            # (tgt_sent_len, batch_size)
            tgt_words_mask = (tgt_sents_var != self.vocab.tgt['<pad>']).float()

            # (tgt_sent_len - 1, batch_size)
            # tgt_words_log_prob : shape: (T-1, B, |V|)
            # 디코더가 각 시점마다 모든 단어에 대해 예측한 log-prob
            # tgt_sents_var[1:] : shape: (T-1, B)
            # 정답 토큰 id, <s> 다음부터 시작하는 gold target
            # t에서 모델이 정답 단어에 부여한 log-prob만 추출
            tgt_gold_words_log_prob = torch.gather(tgt_words_log_prob, index=tgt_sents_var[1:].unsqueeze(-1), dim=-1).squeeze(-1) * tgt_words_mask[1:]

        # (batch_size)
        # 각 문장에서 모델이 정답 단어의 부여한 확률 (배치사이즈의 1차원 벡터)
        scores = tgt_gold_words_log_prob.sum(dim=0)

        return scores

    def get_attention_mask(self, src_encodings: torch.Tensor, src_sents_len: List[int]) -> torch.Tensor:
        src_sent_masks = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(src_sents_len):
            src_sent_masks[e_id, src_len:] = 1

        return src_sent_masks.to(self.device)

    def encode(self, src_sents_var: torch.Tensor, src_sent_lens: List[int]) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: list of tuples (h, c) for each decoder layer
        """

        # (src_sent_len, batch_size, embed_size)
        src_word_embeds = self.src_embed(src_sents_var)
        packed_src_embed = pack_padded_sequence(src_word_embeds, src_sent_lens)

        # src_encodings: (src_sent_len, batch_size, hidden_size)
        # last_state, last_cell: (num_layers * 2, batch_size, hidden_size)
        # src_encodings : encoder output (last_state : h_n , last_cell : kwargs()
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings)

        # (batch_size, src_sent_len, hidden_size)
        src_encodings = src_encodings.permute(1, 0, 2)
        
        batch_size = src_encodings.size(0)

        # Handle multi-layer encoder state
        # Reshape to (num_layers, directions, batch, hidden)
        # last_cell = last_cell.view(self.num_layers, 1, batch_size, self.hidden_size) # uni-direct
        
        # dec_init_cell = self.decoder_cell_init(torch.cat([last_layer_cell[0], last_layer_cell[1]], dim=1))
        dec_init_cell = last_cell  # uni-direct
        # dec_init_state = torch.tanh(dec_init_cell)
        dec_init_state = last_state

        # Replicate the initial state for all decoder layers

        if self.use_all_layer_hiddenstates :
            decoder_init_vec = [(dec_init_state[i], dec_init_cell[i]) for i in range(self.num_layers)]
        else :
            decoder_init_vec = [(dec_init_state[-1], dec_init_cell[-1]) for i in range(self.num_layers)]

        return src_encodings, decoder_init_vec

    def decode(self, src_encodings: torch.Tensor, src_sent_masks: torch.Tensor,
               decoder_init_vec: List[Tuple[torch.Tensor, torch.Tensor]], tgt_sents_var: torch.Tensor, src_len_tensor: torch.Tensor = None) -> torch.Tensor:
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens
        """

        # (batch_size, src_sent_len, hidden_size)
        if self.use_attention:
            src_encoding_att_linear = self.att_src_linear(src_encodings)
        else:
            src_encoding_att_linear = None

        batch_size = src_encodings.size(0)

        # initialize the attentional vector
        att_tm1 = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # (tgt_sent_len, batch_size, embed_size)
        tgt_word_embeds = self.tgt_embed(tgt_sents_var)

        # h_tm1 is now a list of (h, c) tuples
        # h_tm1 는 h, c 정보를 모두 가짐
        h_tm1 = decoder_init_vec

        att_ves = []

        # start from y_0=`<s>`, iterate until y_{T-1}
        for t, y_tm1_embed in enumerate(tgt_word_embeds.split(split_size=1)):
            y_tm1_embed = y_tm1_embed.squeeze(0)
            if self.input_feed:
                # input feeding: concate y_tm1 and previous attentional vector
                # (batch_size, hidden_size + embed_size)
                x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
            else:
                x = y_tm1_embed
            # src_encodings, src_sent_masks 은 어텐션 사용 시 이용됨
            
            new_states, att_t, alpha_t = self.step(x, h_tm1, src_encodings, src_encoding_att_linear, src_sent_masks, t=t, src_len_tensor=src_len_tensor)

            att_tm1 = att_t
            h_tm1 = new_states
            att_ves.append(att_t)

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        att_ves = torch.stack(att_ves)

        return att_ves

    def step(self, x: torch.Tensor,
             h_tm1_list: List[Tuple[torch.Tensor, torch.Tensor]],
             src_encodings: torch.Tensor, src_encoding_att_linear: torch.Tensor, src_sent_masks: torch.Tensor, t: int = 0, src_len_tensor: torch.Tensor = None) -> Tuple[List[Tuple], torch.Tensor, torch.Tensor]:
        
        current_input = x
        new_states = []

        for i, cell in enumerate(self.decoder_lstms):
            h_prev, c_prev = h_tm1_list[i]
            h_curr, c_curr = cell(current_input, (h_prev, c_prev))
            new_states.append((h_curr, c_curr))
            
            current_input = h_curr
            # Apply dropout between LSTM layers (but not after the last one, as per convention, or applied consistently)
            if i < self.num_layers - 1:
                current_input = self.dropout(current_input)
        
        # Use the hidden state of the top layer for attention
        # 수정 필요할수도
        h_t_top = new_states[-1][0]

        if self.use_attention:
            ctx_t, alpha_t = self.apply_attention(h_t_top, src_encodings, src_encoding_att_linear, src_sent_masks, t, src_len_tensor=src_len_tensor)
            att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t_top, ctx_t], 1)))  # E.q. (5)
            att_t = self.dropout(att_t)
        else:
            att_t = self.dropout(h_t_top)
            alpha_t = None

        return new_states, att_t, alpha_t

    def apply_attention(self, h_t: torch.Tensor, src_encoding: torch.Tensor, src_encoding_att_linear: torch.Tensor,
                           mask: torch.Tensor=None, t: int=0, src_len_tensor: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_t: (batch_size, hidden_size)
        # src_encoding: (batch_size, src_sent_len, hidden_size)
        
        batch_size, src_len, _ = src_encoding.size()

        # 1. Score Calculation
        if self.att_score == 'dot':
            att_weight = torch.bmm(src_encoding, h_t.unsqueeze(2)).squeeze(2)
        elif self.att_score == 'general':
            # Ensure att_Wa is used. In global attention, pre-computation is efficient.
            # If src_encoding_att_linear is passed (precomputed W_a * h_s), we use it.
            # Otherwise we compute it.
            if src_encoding_att_linear is not None:
                att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
            else:
                # Compute on the fly (e.g. for local if not precomputed or distinct params)
                # For general, it's efficient to precompute.
                # W_a h_s
                temp = self.att_Wa(src_encoding) 
                att_weight = torch.bmm(temp, h_t.unsqueeze(2)).squeeze(2)
        elif self.att_score == 'location':
            # Location-based score: a_t = softmax(Wa(h_t))
            # Wa is (hidden, 100) -> output (batch, 100)
            raw_scores = self.att_Wa(h_t) # (B, 100)
            # Slice to current src_len (which is batch max len in Tensor)
            att_weight = raw_scores[:, :src_len]
        else:
            # Default fallback for safety
            att_weight = torch.bmm(src_encoding, h_t.unsqueeze(2)).squeeze(2)

        # 2. Local Attention Alignment (p_t)
        if self.att_type == 'global':
            # Global: All positions
            pass
        elif self.att_type == 'local-m':
            # Monotonic: p_t = t
            p_t = torch.tensor([t] * batch_size, device=self.device, dtype=torch.float)
        elif self.att_type == 'local-p':
            # Predictive: p_t = S * sigmoid(v_p^T tanh(W_p h_t))
            # S = actual sentence length for each batch item
            x = torch.tanh(self.att_Wp(h_t))
            score_p = torch.sigmoid(self.att_vp(x)).squeeze(1) # (B, )
            if src_len_tensor is not None:
                p_t = score_p * src_len_tensor # (B, )
            else:
                p_t = score_p * src_len # fallback if not provided

        # 3. Masking
        # (1) Padding Mask (original 'mask' arg)
        if mask is not None:
            att_weight.data.masked_fill_(mask.bool(), -float('inf'))

        # (2) Window Masking (Local only)
        if self.att_type in ['local-m', 'local-p']:
            # Create window mask
            # window: [p_t - D, p_t + D]
            # p_t is (B, )
            # We need to mask positions j where j < p_t - D or j > p_t + D
            # positions: (1, L)
            positions = torch.arange(src_len, device=self.device).unsqueeze(0).float() # (1, L)
            
            p_t_expanded = p_t.unsqueeze(1) # (B, 1)
            D = self.window_size
            
            # Mask condition: |j - p_t| > D
            # => (j < p_t - D) or (j > p_t + D)
            start = p_t_expanded - D
            end = p_t_expanded + D
            
            local_mask = (positions < start) | (positions > end)
            att_weight.data.masked_fill_(local_mask, -float('inf'))
        """
        print("att_type: ", self.att_type)
        print("att_weight.shape: ", att_weight.shape)
        """
        # 4. Mask Checks to prevent NaN
        # Check if any row is entirely -inf
        all_masked = torch.all(att_weight == -float('inf'), dim=-1)
        if all_masked.any():
            att_weight[all_masked, 0] = 0.0

        # 4. Softmax
        softmaxed_att_weight = F.softmax(att_weight, dim=-1) # alpha_t

        # 5. Gaussian Weighting (Local-p only)
        # Apply AFTER softmax as per prompt: "Softmax 결과인 alpha_t에 요소별 곱"
        if self.att_type == 'local-p':
            # Gaussian: exp( - (s - p_t)^2 / (2 * sigma^2) )
            # sigma = D / 2
            positions = torch.arange(src_len, device=self.device).float()
            sigma = self.window_size / 2.0
            
            # (B, L)
            numerator = (positions.unsqueeze(0) - p_t.unsqueeze(1)) ** 2
            gauss_weight = torch.exp(-numerator / (2 * sigma ** 2))
            
            softmaxed_att_weight = softmaxed_att_weight * gauss_weight

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(softmaxed_att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, softmaxed_att_weight


    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """
        Given a single source sentence, perform beam search
        """

        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_len = len(src_sent)
        src_encodings, dec_init_vec = self.encode(src_sents_var, [src_len])

        # Prepare src_len_tensor for beam search
        src_len_tensor = torch.tensor([src_len], dtype=torch.float, device=self.device)

        if self.use_attention:
            src_encodings_att_linear = self.att_src_linear(src_encodings)
        else:
            src_encodings_att_linear = None

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        # attention history for each hypothesis
        att_histories = [[]]
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            if self.use_attention:
                exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                               src_encodings_att_linear.size(1),
                                                                               src_encodings_att_linear.size(2))
            else:
                exp_src_encodings_att_linear = None
            
            # Expand src_len_tensor
            exp_src_len_tensor = src_len_tensor.expand(hyp_num)

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_tm1_embed = self.tgt_embed(y_tm1)

            if self.input_feed:
                x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
            else:
                x = y_tm1_embed

            new_states, att_t, alpha_t = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, src_sent_masks=None, t=t, src_len_tensor=exp_src_len_tensor)

            if alpha_t is not None:
                # (batch_size)
                att_idx = alpha_t.argmax(dim=-1).cpu().numpy().tolist()
            else:
                att_idx = [0] * len(hypotheses)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.readout(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            # FIX: Use integer division // for indexing
            prev_hyp_ids = top_cand_hyp_pos // len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            new_att_histories = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]

                current_att_idx = att_idx[prev_hyp_id]
                new_att_history = att_histories[prev_hyp_id] + [current_att_idx]

                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score,
                                                           attention_history=new_att_history))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)
                    new_att_histories.append(new_att_history)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            
            # Reconstruct h_tm1 for the next step based on live hypotheses
            h_tm1 = []
            for layer_h, layer_c in new_states:
                h_tm1.append((layer_h[live_hyp_ids], layer_c[live_hyp_ids]))
            
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)
            att_histories = new_att_histories

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item(),
                                                   attention_history=att_histories[0]))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    def sample(self, src_sents: List[List[str]], sample_size=5, max_decoding_time_step=100) -> List[Hypothesis]:
        """
        Given a batched list of source sentences, randomly sample hypotheses from the model distribution p(y|x)
        """

        src_sents_var = self.vocab.src.to_input_tensor(src_sents, self.device)

        src_sents_len = [len(sent) for sent in src_sents]
        src_encodings, dec_init_vec = self.encode(src_sents_var, src_sents_len)
        
        # Prepare src_len_tensor
        src_len_tensor = torch.tensor(src_sents_len, dtype=torch.float, device=self.device)
        src_len_tensor = src_len_tensor.repeat(sample_size)

        if self.use_attention:
            src_encodings_att_linear = self.att_src_linear(src_encodings)
        else:
            src_encodings_att_linear = None

        h_tm1 = dec_init_vec

        batch_size = len(src_sents)
        total_sample_size = sample_size * len(src_sents)

        # (total_sample_size, max_src_len, src_encoding_size)
        src_encodings = src_encodings.repeat(sample_size, 1, 1)

        if self.use_attention:
            src_encodings_att_linear = src_encodings_att_linear.repeat(sample_size, 1, 1)
        else:
            src_encodings_att_linear = None

        src_sent_masks = self.get_attention_mask(src_encodings, [len(sent) for _ in range(sample_size) for sent in src_sents])

        # Replicate initial states for all samples
        h_tm1 = []
        for h, c in dec_init_vec:
            h_tm1.append((h[0].repeat(sample_size, 1), c[0].repeat(sample_size, 1)))

        att_tm1 = torch.zeros(total_sample_size, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']
        sample_ends = torch.zeros(total_sample_size, dtype=torch.uint8, device=self.device)
        sample_scores = torch.zeros(total_sample_size, device=self.device)

        samples = [torch.tensor([self.vocab.tgt['<s>']] * total_sample_size, dtype=torch.long, device=self.device)]

        t = 0
        while t < max_decoding_time_step:
            t += 1

            y_tm1 = samples[-1]

            y_tm1_embed = self.tgt_embed(y_tm1)

            if self.input_feed:
                x = torch.cat([y_tm1_embed, att_tm1], 1)
            else:
                x = y_tm1_embed

            new_states, att_t, alpha_t = self.step(x, h_tm1,
                                                      src_encodings, src_encodings_att_linear,
                                                      src_sent_masks=src_sent_masks, t=t, src_len_tensor=src_len_tensor)
            # difference with my code
            # manually compute negative log likelyhood loss instead of using torch nn modules
            # probabilities over target words
            # forward 함수에서는 F.log_softmax -> manual NLL 계산
            # sample 함수에서는 F.softmax -> torch.log , manual NLL계산
            p_t = F.softmax(self.readout(att_t), dim=-1)
            log_p_t = torch.log(p_t)

            # (total_sample_size)
            # 확률 분포에 따라 문장을 **무작위 생성(Random Sampling)
            # p_t는 현재 시점($t$)에서 모델이 예측한 단어별 확률값들(Softmax 결과)을 담고 있습니다. multinomial은 이 확률에 비례하여 단어를 뽑
            y_t = torch.multinomial(p_t, num_samples=1)
            log_p_y_t = torch.gather(log_p_t, 1, y_t).squeeze(1)
            y_t = y_t.squeeze(1)

            samples.append(y_t)
            # 문장이 끝났는지(End of Sentence, EOS)를 추적
            # |= (Bitwise OR 연산)
            sample_ends |= torch.eq(y_t, eos_id).byte()
            # (1. - sample_ends.float())는 이미 끝난 문장의 점수는 더 이상 합산하지 않기 위한 필터링
            sample_scores = sample_scores + log_p_y_t * (1. - sample_ends.float())
            # sample_scores은 배치 내 각 문장의 각 단어들의 음의 로그 우도의 총합 (한 문장의 스코어)

            if torch.all(sample_ends):
                break

            att_tm1 = att_t
            h_tm1 = new_states

        _completed_samples = [[[] for _1 in range(sample_size)] for _2 in range(batch_size)]
        for t, y_t in enumerate(samples):
            for i, sampled_word_id in enumerate(y_t):
                sampled_word_id = sampled_word_id.cpu().item()
                src_sent_id = i % batch_size
                sample_id = i // batch_size

                if t == 0 or _completed_samples[src_sent_id][sample_id][-1] != eos_id:
                    _completed_samples[src_sent_id][sample_id].append(sampled_word_id)

        completed_samples = [[None for _1 in range(sample_size)] for _2 in range(batch_size)]
        for src_sent_id in range(batch_size):
            for sample_id in range(sample_size):
                offset = sample_id * batch_size + src_sent_id
                # Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
                # completed_samples[i][j]는 **"i번째 소스 문장에 대해 j번째로 생성된 번역 결과를 담음음
                hyp = Hypothesis(value=self.vocab.tgt.indices2words(_completed_samples[src_sent_id][sample_id])[:-1],
                                 score=sample_scores[offset].item(),
                                 attention_history=[])
                completed_samples[src_sent_id][sample_id] = hyp

        return completed_samples

    @staticmethod
    def load(model_path: str, attention_config):
        # FIX: weights_only=False for PyTorch 2.6+ compatibility
        params = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        args = params['args']
        print(args)
        # backward compatibility
        args['use_attention'] = args.get('use_attention', True)
        args['num_layers'] = args.get('num_layers', 1) # Default to 1 if loading old model
        model = NMT(vocab=params['vocab'], **args)

        if args['use_attention'] and attention_config:
            att_type = attention_config['att_type']
            att_score = attention_config['att_score']
            window_size = attention_config['window_size']
            model.set_attention_config(att_type, att_score, window_size)
        model.load_state_dict(params['state_dict'])
        print(model)

        return model

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate,
                         input_feed=self.input_feed, label_smoothing=self.label_smoothing, 
                         use_attention=self.use_attention, num_layers=self.num_layers),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


def debug_log_reverse(name, original_src_sents, reversed_src_sents, vocab, device):
    """
    Log original and reversed source sentence tensors for debugging.
    Only logs the first sentence of the batch.
    """
    print(f"--- {name} Debugging ---", file=sys.stderr)
    
    # Take the first sentence
    orig_sent = original_src_sents[0]
    rev_sent = reversed_src_sents[0]
    
    print(f"Original Text: {orig_sent}", file=sys.stderr)
    print(f"Reversed Text: {rev_sent}", file=sys.stderr)
    
    # Convert to tensor
    # We wrap in a list because to_input_tensor expects a batch
    orig_tensor = vocab.src.to_input_tensor([orig_sent], device)
    rev_tensor = vocab.src.to_input_tensor([rev_sent], device)
    
    print(f"Original Source Sentence Tensor:\n{orig_tensor}", file=sys.stderr)
    print(f"Reversed Source Sentence Tensor:\n{rev_tensor}", file=sys.stderr)
    print("-------------------------", file=sys.stderr)


def evaluate_ppl(model, dev_data, batch_size=32, reverse=False):

    """
    Evaluate perplexity on dev sentences

    Args:
        dev_data: a list of dev sentences
        batch_size: batch size

    Returns:
        ppl: the perplexity on dev sentences
    """

    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # you may want to wrap the following code using a context manager provided
    # by the NN library to signal the backend to not to keep gradient information
    # e.g., `torch.no_grad()`

    with torch.no_grad():
        first_batch = True
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            if reverse:
                reversed_src_sents = [s[::-1] for s in src_sents]
                #if first_batch:
                #    #debug_log_reverse("Evaluation", src_sents, reversed_src_sents, model.vocab, model.device)
                #    first_batch = False
                src_sents = reversed_src_sents
            
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """

    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train(args: Dict):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = Vocab.load(args['--vocab'])
    
    # Determine whether to use attention based on the flag
    use_attention = not args['--no-attention']
    
    # Parse num_layers
    num_layers = int(args['--num-layers'])
    lr_decay_start = int(args['--lr-decay-start'])
    use_all_layer_hiddenstates = args['--use-all-layer-hiddenstates']
    reverse = args['--reverse']

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                input_feed=args['--input-feed'],
                label_smoothing=float(args['--label-smoothing']),
                use_attention=use_attention,
                num_layers=num_layers,
                vocab=vocab,
                use_all_layer_hiddenstates=use_all_layer_hiddenstates)
    
    # Configure attention
    if use_attention:
        att_type = args.get('--att-type', 'global')
        att_score = args.get('--att-score', 'dot')
        window_size = int(args.get('--window-size', 10))
        model.set_attention_config(att_type, att_score, window_size)

    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)
    # 타겟 어휘 마스킹
    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))
    optimizer = torch.optim.SGD(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            if reverse:
                reversed_src_sents = [s[::-1] for s in src_sents]
                # Log only for the very first batch of the first epoch (or every epoch? Requirement says "first batch/sentence of each phase")
                # Let's log once per execution or once per epoch? "학습, 평가, 추론 각각의 첫 번째 배치/문자에 대해"
                # Interpreting as "First batch encountered in this run" for training.
                # However, since training runs for many epochs, maybe once per training run is enough.
                # But to be safe and visible, I'll do it once at the start of training loop (flag).
                #if train_iter == 1:
                     #debug_log_reverse("Training", src_sents, reversed_src_sents, vocab, device)
                src_sents = reversed_src_sents

            optimizer.zero_grad()

            batch_size = len(src_sents)

            # (batch_size)
            example_losses = -model(src_sents, tgt_sents)
            batch_loss = example_losses.sum()
            # 한 배치 내 모든 문장의 손실 평균 : loss
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), clip_grad)

            optimizer.step()
            # mark till here
            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            with torch.no_grad():
                if train_iter % valid_niter == 0:
                    print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),cum_examples), file=sys.stderr)

                    cum_loss = cum_examples = cum_tgt_words = 0.
                    valid_num += 1

                    print('begin validation ...', file=sys.stderr)

                    # compute dev. ppl and bleu
                    dev_ppl = evaluate_ppl(model, dev_data, batch_size=128, reverse=reverse)   # dev batch size can be a bit larger
                    valid_metric = -dev_ppl

                    print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)
                    
                    is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                    hist_valid_scores.append(valid_metric)
                    if is_better:
                        patience = 0
                        print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                        model.save(model_save_path)

                        # also save the optimizers' state
                        torch.save(optimizer.state_dict(), model_save_path + '.optim')
                    elif patience < int(args['--patience']):
                        patience += 1
                        print('hit patience %d' % patience, file=sys.stderr)
                        '''
                        if patience == int(args['--patience']):
                            num_trial += 1
                            print('hit #%d trial' % num_trial, file=sys.stderr)
                            if num_trial == int(args['--max-num-trial']):
                                print('early stop!', file=sys.stderr)
                                exit(0)

                            # decay lr, and restore from previously best checkpoint
                            lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                            # load model
                            # FIX: weights_only=False for PyTorch 2.6+ compatibility
                            params = torch.load(model_save_path, map_location=lambda storage, loc: storage, weights_only=False)
                            model.load_state_dict(params['state_dict'])
                            model = model.to(device)

                            print('restore parameters of the optimizers', file=sys.stderr)
                            # FIX: weights_only=False for PyTorch 2.6+ compatibility
                            optimizer.load_state_dict(torch.load(model_save_path + '.optim', weights_only=False))

                            # set new lr
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr

                            # reset patience
                            patience = 0
                        '''
        if epoch >= lr_decay_start:
            # 현재 학습률에 0.5(lr-decay)를 곱함
            lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
            
            print('epoch %d 종료: 학습률을 %f로 감소시킵니다.' % (epoch, lr), file=sys.stderr)
    
            # 옵티마이저에 새로운 학습률 적용
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # 최대 에포크 도달 시 종료
        if epoch == int(args['--max-epoch']):
            print('reached maximum number of epochs!', file=sys.stderr)
            exit(0)

def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int, reverse: bool = False) -> List[List[Hypothesis]]:
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        first_sent = True
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            if reverse:
                reversed_src_sent = src_sent[::-1]
                #if first_sent:
                #     #debug_log_reverse("Inference", [src_sent], [reversed_src_sent], model.vocab, model.device)
                #     first_sent = False
                src_sent = reversed_src_sent
            
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """

    print(f"load test source sentences from [{args['TEST_SOURCE_FILE']}]", file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print(f"load test target sentences from [{args['TEST_TARGET_FILE']}]", file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    '''
    attention_config :
    --att-type=<str>                        attention type (global, local-m, local-p) [default: global]
    --att-score=<str>                       attention score function (dot, general, location) [default: dot]
    --window-size=<int>                     window size D for local attention [default: 10]
    '''
    if not args['--no-attention']:
        attention_config={
            "att_type": args['--att-type'],
            "att_score": args['--att-score'],
            "window_size": int(args['--window-size'])
        }
    else:
        attention_config=None
    model = NMT.load(args['MODEL_PATH'],attention_config)    

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))    # Configure attention

    unk_replace = args.get('--unk-replace', False)
    unk_dict = {}
    if unk_replace:
        unk_dict_path = args.get('--unk-dict', 'data/dict.en-de.json')
        try:
            with open(unk_dict_path, 'r') as f:
                unk_dict = json.load(f)
        except FileNotFoundError:
            print(f"[Warning] Unknown dictionary file not found at {unk_dict_path}. Replacement will use identity.", file=sys.stderr)

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']),
                             reverse=args['--reverse'])

    if unk_replace:
        for i, (src_sent, hyps) in enumerate(zip(test_data_src, hypotheses)):
            top_hyp = hyps[0]
            new_tokens = []
            replaced = False

            if args['--reverse']:
                effective_src_sent = src_sent[::-1]
            else:
                effective_src_sent = src_sent

            for t_val, att_idx in zip(top_hyp.value, top_hyp.attention_history):
                if t_val == '<unk>':
                    if att_idx < len(effective_src_sent):
                        src_word = effective_src_sent[att_idx]
                        replacement = unk_dict.get(src_word.lower(), src_word)
                        print(f"[UNK Replace] Target <unk> (src_pos: {att_idx}, word: '{src_word}') -> Replaced with: '{replacement}'", file=sys.stderr)
                        new_tokens.append(replacement)
                        replaced = True
                    else:
                        new_tokens.append(t_val)
                else:
                    new_tokens.append(t_val)
            
            if replaced:
                original_sent = ' '.join(top_hyp.value)
                final_sent = ' '.join(new_tokens)
                print(f"Sentence {i} Summary:\n Original: {original_sent}\n Final   : {final_sent}", file=sys.stderr)
                hyps[0] = top_hyp._replace(value=new_tokens)

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)
    print(args)
    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid run mode')


if __name__ == '__main__':
    main()
