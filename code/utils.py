# coding: UTF-8
import logging
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics
import transformers
from torch.utils.data import DataLoader
import json
import collections
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
# from nltk.translate.nist_score import corpus_nist
from nltk.translate.nist_score import sentence_nist
import math
from nltk.util import ngrams 
from rouge import Rouge


def distinct_n_sentence_level(sentence, n):
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)



def train(model, optimizer, scheduler, loader):
    model.train()
    total_iter = 0
    total_loss = 0.

    for inputs in loader:
    # for inputs in tqdm(loader):
        model.zero_grad()
        output = model(**inputs)
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_iter += 1
        total_loss += loss.data

    avg_loss = total_loss / total_iter
    logging.info(f'[Train_Loss]: {avg_loss}')


def contrast_train(model, optimizer, scheduler, loader, args):
    model.train()
    total_iter = 0
    total_loss = 0.
    total_contrastive_loss = 0.
    total_masked_lm_loss = 0.

    for inputs in loader:
        model.zero_grad()
        output = model.contrast_with_generation_forward(**inputs)
        
        if 'Contrast' in args.mode and 'Generation' in args.mode:
            loss = output.loss
        elif 'Contrast' in args.mode:
            loss = output.contrastive_loss
        else:
            loss = output.masked_lm_loss
        # loss = output.contrastive_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_iter += 1
        total_loss += loss.data
        total_contrastive_loss += output.contrastive_loss.data
        total_masked_lm_loss += output.masked_lm_loss.data

    avg_loss = total_loss / total_iter
    avg_contrastive_loss = total_contrastive_loss / total_iter
    avg_masked_lm_loss = total_masked_lm_loss / total_iter
    logging.info(f'[Train_Loss]: {avg_loss}, contrastive_loss: {avg_contrastive_loss}, masked_lm_loss: {avg_masked_lm_loss}')
    # logging.info(f'[Train_Loss]: {avg_loss}')


def test(model, loader, tokenizer, filename=None):
    model.eval() 

    cnt = 0
    bleu1 = 0.
    bleu2 = 0.
    bleu3 = 0.
    bleu4 = 0.
    rouge1 = 0.
    rouge2 = 0.
    rougeL = 0.
    meteor =0.
    nist1 = 0.
    nist2 = 0.
    nist3 = 0.
    nist4 = 0.

    rouge = Rouge()
    
    all_predicts = []

    if filename is not None:
        with open(filename, 'w', encoding='utf-8') as f:
            with torch.no_grad():
                for inputs in loader:
                # for inputs in tqdm(loader):
                    model.zero_grad()
                    # inputs['mode'] = mode
                    input_ids = inputs['input_ids']
                    inputs.pop('input_ids')
                    attention_mask = inputs['attention_mask']
                    inputs.pop('attention_mask')
                    labels = inputs['labels']
                    inputs.pop('labels')
                    outputs = model.generate(input_ids, attention_mask=attention_mask, do_sample=False, num_beams=3 ,max_length=32, **inputs)
   
                    predicts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    original = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    for i in range(len(labels)):
                        label = original[i].split()
             
                        predict = predicts[i].split()
                        all_predicts += predict
                        f.write(' '.join(label) + '\n')
                        f.write(' '.join(predict) + '\n')

                        min_len = min(len(predict), len(label))
                        lens = min(min_len, 4)
                        if lens == 0:
                            continue

                        if not (''.join(predict)).strip():
                            continue
                        
                        temp_predict = ' '.join(predict)
                        temp_label = ' '.join(label)

                        try:
                            
                            rouge_score = rouge.get_scores(temp_predict, temp_label)
                            rouge1 += rouge_score[0]["rouge-1"]['r']
                            rouge2 += rouge_score[0]["rouge-2"]['r']
                            rougeL += rouge_score[0]["rouge-l"]['f']

                            label = [label]
                            if lens >= 1:
                                bleu1 += sentence_bleu(label, predict, weights=(1, 0, 0, 0))
                                nist1 += sentence_nist(label, predict, 1)
                            if lens >= 2:
                                bleu2 += sentence_bleu(label, predict, weights=(0.5, 0.5, 0, 0))
                                nist2 += sentence_nist(label, predict, 2)
                            if lens >= 3:
                                bleu3 += sentence_bleu(label, predict, weights=(0.333, 0.333, 0.333, 0))
                                nist3 += sentence_nist(label, predict, 3)
                            if lens >= 4:
                                bleu4 += sentence_bleu(label, predict, weights=(0.25, 0.25, 0.25, 0.25))
                                nist4 += sentence_nist(label, predict, 4)
                            
                            meteor += meteor_score(label, predict)

                            cnt += 1
                        except:
                            print(temp_predict)
                            print(temp_label)
    else:
        for inputs in loader:
            model.zero_grad()
            input_ids = inputs['input_ids']
            inputs.pop('input_ids')
            attention_mask = inputs['attention_mask']
            inputs.pop('attention_mask')
            labels = inputs['labels']
            inputs.pop('labels')
            outputs = model.generate(input_ids, attention_mask=attention_mask, do_sample=False, num_beams=3 ,max_length=32, **inputs)
            predicts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            original = tokenizer.batch_decode(labels, skip_special_tokens=True)
            for i in range(len(labels)):
                label = original[i].split()
                predict = predicts[i].split()
                all_predicts += predict

                temp_predict = ' '.join(predict)
                temp_label = ' '.join(label)

                min_len = min(len(predict), len(label))
                lens = min(min_len, 4)
                if lens == 0:
                    continue

                if not (''.join(predict)).strip():
                    continue
                
                
                try:
                    rouge_score = rouge.get_scores(temp_predict, temp_label)
                    rouge1 += rouge_score[0]["rouge-1"]['r']
                    rouge2 += rouge_score[0]["rouge-2"]['r']
                    rougeL += rouge_score[0]["rouge-l"]['f']

                    label = [label]
                    if lens >= 1:
                        bleu1 += sentence_bleu(label, predict, weights=(1, 0, 0, 0))
                        nist1 += sentence_nist(label, predict, 1)
                    if lens >= 2:
                        bleu2 += sentence_bleu(label, predict, weights=(0.5, 0.5, 0, 0))
                        nist2 += sentence_nist(label, predict, 2)
                    if lens >= 3:
                        bleu3 += sentence_bleu(label, predict, weights=(0.333, 0.333, 0.333, 0))
                        nist3 += sentence_nist(label, predict, 3)
                    if lens >= 4:
                        bleu4 += sentence_bleu(label, predict, weights=(0.25, 0.25, 0.25, 0.25))
                        nist4 += sentence_nist(label, predict, 4)
                    
                    meteor += meteor_score(label, predict)

                    cnt += 1
                except:
                    print(temp_predict)
                    print(temp_label)

    dist1 = distinct_n_sentence_level(all_predicts, 1)
    dist2 = distinct_n_sentence_level(all_predicts, 2)

    bleu1 /= cnt
    bleu2 /= cnt
    bleu3 /= cnt
    bleu4 /= cnt
    rouge1 /= cnt
    rouge2 /= cnt
    rougeL /= cnt
    meteor /= cnt
    nist1 /= cnt
    nist2 /= cnt
    nist3 /= cnt
    nist4 /= cnt
    
    bleu1 *= 100
    bleu2 *= 100
    bleu3 *= 100
    bleu4 *= 100
    dist1 *= 100
    dist2 *= 100
    rouge1 *= 100
    rouge2 *= 100
    rougeL *= 100
    meteor *= 100
    nist1 *= 100
    nist2 *= 100
    nist3 *= 100
    nist4 *= 100

    return bleu1, bleu2, bleu3, bleu4, dist1, dist2, rouge1, rouge2, rougeL, meteor, nist1, nist2, nist3, nist4