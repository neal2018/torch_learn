#!/usr/bin/env python
# coding: utf-8

# In[1]:
import math
import os
import random
import sys
from collections import Counter

import jieba
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


def load_data(file):
    en = []
    cn = []
    with open(file, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split("\t")
            en.append(["BOS"] + nltk.word_tokenize(line[0].lower()) + ["EOS"])
            cn.append(
                ["BOS"] + [c for c in jieba.cut(line[1])] + ["EOS"]
            )  # jieba.cut generator to list
    return en, cn


# In[3]:


def build_dict(sentences, max_words=50000):
    UNK_IDX = 0
    PAD_IDX = 1
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s]+=1
    ls = word_count.most_common(max_words)
    word_dict = {w[0]:index+2 for index, w in enumerate(ls)}
    word_dict["UNK"] = UNK_IDX
    word_dict["PAD"] = PAD_IDX
    total_words = len(ls) + 2
    return word_dict, total_words


# In[4]:


def encode(en_sens, cn_sens, en_dict, cn_dict, sort_by_len=True):
    """
    word to number
    """
    out_en_sens = [[en_dict.get(w, 0) for w in en_sen] for en_sen in en_sens]
    out_cn_sens = [[cn_dict.get(w, 0) for w in cn_sen] for cn_sen in cn_sens]
    
    if sort_by_len:
        sorted_index = sorted(range(len(out_en_sens)), key=lambda x: len(out_en_sens[x]))
        out_en_sens = [out_en_sens[i] for i in sorted_index]
        out_cn_sens = [out_cn_sens[i] for i in sorted_index]
    return out_en_sens, out_cn_sens


# In[5]:


def get_mini_batches(n, sz, shuffle=True):
    """
    seperate range(n) into batches with size of `sz`
    """
    minibatches=[np.arange(idx, min(idx+sz, n)) for idx in range(0, n, sz)]
    if shuffle:
        np.random.shuffle(minibatches)
    return minibatches


# In[13]:


def prepare_data(seqs):
    """
    pading seqs to a matrix
    """
    lengths = torch.tensor([len(seq) for seq in seqs])
    x = [torch.tensor(seq) for seq in seqs]
    x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True)
    return x_padded, lengths


# In[14]:


def gen_examples(en_sens, cn_sens, minibatch_size):
    minibatches = get_mini_batches(len(en_sens), minibatch_size)
    all_ex=[]
    for minibatch in minibatches:
        mb_en_sents = [en_sens[t] for t in minibatch]
        mb_cn_sents = [cn_sens[t] for t in minibatch]
        mb_x, mb_x_len = prepare_data(mb_en_sents)
        mb_y, mb_y_len = prepare_data(mb_cn_sents)
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))
    return all_ex


# ### without attention

# In[30]:


class PlainEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, lengths):
        embedded = self.dropout(self.embed(x))
        # mark the end of the sentence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        return out, hid[[-1]]


# In[40]:


class PlainDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, y, y_lengths, hid):
        embedded = self.dropout(self.embed(y))

        packed_seq = nn.utils.rnn.pack_padded_sequence(embedded, y_lengths, batch_first=True, enforce_sorted=False)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        output = F.log_softmax(self.out(unpacked), -1)
        
        return output, hid


# In[41]:


class PlainSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid = self.decoder(y, y_lengths, hid)
        return output, None
    
    def translate(self, x, x_lengths, y, max_len=10):
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_len):
            output, hid = self.decoder(y, torch.ones(batch_size).long().to(y.device), hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
        return torch.cat(preds, 1), None


# In[42]:


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, target, mask):
        # x: batch_size * seq_len * vocab_size
        x = x.contiguous().view(-1, x.size(2))
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = -x.gather(1, target) * mask
        output = torch.sum(output)/torch.sum(mask)
        return output


# In[43]:


def load_model(f):
    def wrapper(model, *args, **kwargs):
        PATH = "./saved_model/no_attention.pth"
        if os.path.exists(PATH):
            model.load_state_dict(torch.load(PATH))
        res = f(model, *args, **kwargs)
        torch.save(model.state_dict(), PATH)
        return res
    return wrapper


# In[44]:


@load_model
def train(model, data, nums_epoches=20):
    for epoch in range(nums_epoches):
        model.train()
        total_num_words = 0
        total_loss = 0
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = mb_x.to(device).long()
            mb_x_len = mb_x_len.to(device).long()
            mb_input = mb_y[:, :-1].to(device).long()
            mb_output = mb_y[:, 1:].to(device).long()
            mb_y_len = (mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            
            if it % 100 == 0:
                print("Epoch", epoch, "iteration", it, "loss", loss.item())

                
        print("Epoch", epoch, "Training loss", total_loss/total_num_words)
        if epoch % 5 == 0:
            evaluate(model, dev_data)


# In[45]:


@load_model
def evaluate(model, data):
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = mb_x.to(device).long()
            mb_x_len = mb_x_len.to(device).long()
            mb_input = mb_y[:, :-1].to(device).long()
            mb_output = mb_y[:, 1:].to(device).long()
            mb_y_len = (mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device).unsqueeze(0) < mb_y_len.unsqueeze(1)
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation loss", total_loss/total_num_words)


# In[46]:

@load_model
def translate_dev(model, i):
    en_sent = " ".join([inv_en_dict[w] for w in dev_en[i]])
    print(en_sent)
    cn_sent = " ".join([inv_cn_dict[w] for w in dev_cn[i]])
    print("".join(cn_sent))

    mb_x = torch.tensor(dev_en[i]).unsqueeze(0).to(device).long()
    mb_x_len = torch.tensor(len(dev_en[i])).unsqueeze(0).to(device).long()
    bos = torch.Tensor([[cn_dict["BOS"]]]).to(device).long()

    translation, attn = model.translate(mb_x, mb_x_len, bos)
    trans1 = [inv_cn_dict[i] for i in translation.cpu().numpy().reshape(-1)]
    trans = []
    for word in trans1:
        if word != "EOS":
            trans.append(word)
        else:
            break
    print("".join(trans))
    print("".join(trans1))


# In[47]:

if __name__ == "__main__":
    train_file = "data/nmt/en-cn/train.txt"
    dev_file = "data/nmt/en-cn/dev.txt"
    train_en, train_cn = load_data(train_file)
    dev_en, dev_cn = load_data(dev_file)

    en_dict, en_total_words = build_dict(train_en)
    cn_dict, cn_total_words = build_dict(train_cn)
    inv_en_dict = {v: k for k, v in en_dict.items()}
    inv_cn_dict = {v: k for k, v in cn_dict.items()}

    train_en, train_cn = encode(train_en, train_cn, en_dict, cn_dict)
    dev_en, dev_cn = encode(dev_en, dev_cn, en_dict, cn_dict)

    batch_size = 64
    train_data = gen_examples(train_en, train_cn, batch_size)
    dev_data = gen_examples(dev_en, dev_cn, batch_size)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    dropout = 0.2
    hidden_size = 100
    encoder = PlainEncoder(vocab_size=en_total_words,
                        hidden_size=hidden_size,
                        dropout=dropout)
    decoder = PlainDecoder(vocab_size=cn_total_words,
                        hidden_size=hidden_size,
                        dropout=dropout)

    model = PlainSeq2Seq(encoder, decoder)
    model = model.to(device)
    loss_fn = LanguageModelCriterion().to(device)

    optimizer = torch.optim.Adam(model.parameters())

    translate_dev(model, 100)