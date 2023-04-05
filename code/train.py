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

from src.data_set import *
from src.models import MMBartForConditionalGeneration
from utils import train, test

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    set_seed
)


name2model = {
    "MMBartForConditionalGeneration": MMBartForConditionalGeneration,
}

name2bertpath = {
    "MMBartForConditionalGeneration": '/home/lqli/project/video_dialog_generation/pretrain/bart-base',
}

def set_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)


def get_args():
    parser = argparse.ArgumentParser(description='The model for disease diagnosis.')
    parser.add_argument('--model_name', default="MMBartForConditionalGeneration", type=str, help='mode')
    parser.add_argument('--mode', default='Text', type=str)
    parser.add_argument('--max_len', default=512, type=int)
    # parser.add_argument('--video_embed_dim', default=2048, type=int)
    parser.add_argument('--video_embed_dim', default=2048, type=int)
    parser.add_argument('--audio_embed_dim', default=128, type=int)
    parser.add_argument('--target_max_len', default=32, type=int)
    parser.add_argument('--max_turns', default=2, type=int, help='max_turns')
    parser.add_argument('--num_epochs', default=10, type=int, help='the epoch of train')
    parser.add_argument('--batch_size', default=32, type=int, help='the batch size of dataset')
    parser.add_argument('--lr', default=1e-5, type=float, help='the learning rate of bert')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='the learning rate of bert')
    parser.add_argument('--warm_up', default=0.01)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=device, type=str, help='the device gpu or cpu')
    parser.add_argument('--seed', default=13, type=int, help='seed')
    parser.add_argument('--output_dir', default='result', type=str)
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--two_roles', action='store_true', default=False)
    parser.add_argument('--alpha', default=1, type=float, help='alpha')


    return parser.parse_args()



def main():
    args = get_args()
    set_logger()
    logging.info(args)
    set_seed(args.seed)

    res = collections.OrderedDict(args.__dict__)
    args.bert_path = name2bertpath[args.model_name]
    args.tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    args.tokenizer.add_tokens(["<video>"], special_tokens=True)
    args.tokenizer.add_tokens(["<audio>"], special_tokens=True)
    config = AutoConfig.from_pretrained(args.bert_path)
    config.video_embed_dim = args.video_embed_dim
    config.audio_embed_dim = args.audio_embed_dim
    config.mode = args.mode
    config.two_roles = args.two_roles
    config.alpha = args.alpha

    model = name2model[args.model_name]
    model = model.from_pretrained(args.bert_path, config=config)
    model.resize_token_embeddings(len(args.tokenizer))
    model.to(args.device)      
    
    if args.two_roles:
        save_dir = f'checkpoint/{args.mode}_{args.max_turns}_two_roles'
    else:
        save_dir = f'checkpoint/{args.mode}_{args.max_turns}'

    if 'Contrast' in args.mode:
        save_dir += '_contrast'
    if 'Generation' in args.mode:
        save_dir += '_generation'
    save_dir += '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.two_roles:
        output_dir = os.path.join(args.output_dir, args.mode + f'_{args.max_turns}' + '_two_roles')
    else:
        output_dir = os.path.join(args.output_dir, args.mode + f'_{args.max_turns}')

    if 'Contrast' in args.mode:
        output_dir += '_contrast'
    if 'Generation' in args.mode:
        output_dir += '_generation'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_txt = os.path.join(output_dir, f'{args.seed}_{args.lr}_result.txt')

    
    logging.info("loading data")
    if args.do_train:
        train_dataset = DialogDataset(args, 'train')
        eval_dataset = DialogDataset(args, 'eval')
    test_dataset = DialogDataset(args, 'test')
    logging.info("finish!")
    
    if args.do_train:
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                collate_fn=train_dataset.collate_fn
                                )
        eval_loader = DataLoader(dataset=eval_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=eval_dataset.collate_fn
                                )
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=test_dataset.collate_fn
                            )


    if args.do_train:
        num_training_steps = len(train_loader) * args.num_epochs
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr, weight_decay=args.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=args.warm_up * num_training_steps,
                                                                num_training_steps=num_training_steps)
    best_epoch = 1
    best_bleu1 = float('-inf')
    bleu1, bleu2, bleu3, bleu4, dist1, dist2, rouge1, rouge2, rougeL = None, None, None, None, None, None, None, None, None
    
    patient = 0

    if args.do_train:
        for i in range(args.num_epochs):
            logging.info(f"epoch {i+1}:")
            train(model, optimizer, scheduler, train_loader)

            if i >= 14:
                bleu1, bleu2, bleu3, bleu4, dist1, dist2, rouge1, rouge2, rougeL, meteor, nist1, nist2, nist3, nist4 = test(model, eval_loader, args.tokenizer)
                logging.info(f'eval: bleu1, bleu2, bleu3, bleu4, dist1, dist2, rouge1, rouge2, rougeL, meteor, nist1, nist2, nist3, nist4: ' +
                            f'{bleu1:.4f}, {bleu2:.4f}, {bleu3:.4f}, {bleu4:.4f}, {dist1:.4f}, {dist2:.4f}, {rouge1:.4f}, {rouge2:.4f}, {rougeL:.4f}, {meteor:.4f}, {nist1:.4f}, {nist2:.4f}, {nist3:.4f}, {nist4:.4f}')
                if bleu1 > best_bleu1:
                    best_epoch = i + 1
                    best_bleu1 = bleu1
                    torch.save(model.state_dict(), f'{save_dir}model_{args.seed}_{args.lr}.bin')
                    logging.info(f'save model in {save_dir}model_{args.seed}_{args.lr}.bin')
                    patient = 0
                else:
                    patient += 1
                    if patient >= 5:
                        break

                bleu1, bleu2, bleu3, bleu4, dist1, dist2, rouge1, rouge2, rougeL, meteor, nist1, nist2, nist3, nist4 = test(model, test_loader, args.tokenizer, result_txt)
                logging.info(f'test: bleu1, bleu2, bleu3, bleu4, dist1, dist2, rouge1, rouge2, rougeL, meteor, nist1, nist2, nist3, nist4: ' + 
                    f'{bleu1:.4f}, {bleu2:.4f}, {bleu3:.4f}, {bleu4:.4f}, {dist1:.4f}, {dist2:.4f}, {rouge1:.4f}, {rouge2:.4f}, {rougeL:.4f}, {meteor:.4f}, {nist1:.4f}, {nist2:.4f}, {nist3:.4f}, {nist4:.4f}')
                
        logging.info(f'best eval epoch: {best_epoch}')
        logging.info(f'load model from {save_dir}model_{args.seed}_{args.lr}.bin')
        model.load_state_dict(torch.load(f'{save_dir}model_{args.seed}_{args.lr}.bin'))

        bleu1, bleu2, bleu3, bleu4, dist1, dist2, rouge1, rouge2, rougeL, meteor, nist1, nist2, nist3, nist4 = test(model, test_loader, args.tokenizer, result_txt)
        logging.info(f'test: bleu1, bleu2, bleu3, bleu4, dist1, dist2, rouge1, rouge2, rougeL, meteor, nist1, nist2, nist3, nist4: ' +
                    f'{bleu1:.4f}, {bleu2:.4f}, {bleu3:.4f}, {bleu4:.4f}, {dist1:.4f}, {dist2:.4f}, {rouge1:.4f}, {rouge2:.4f}, {rougeL:.4f}, {meteor:.4f}, {nist1:.4f}, {nist2:.4f}, {nist3:.4f}, {nist4:.4f}')
    else:
        logging.info(f'load model from {save_dir}model_{args.seed}_{args.lr}.bin')
        model.load_state_dict(torch.load(f'{save_dir}model_{args.seed}_{args.lr}.bin'))

        bleu1, bleu2, bleu3, bleu4, dist1, dist2, rouge1, rouge2, rougeL, meteor, nist1, nist2, nist3, nist4 = test(model, test_loader, args.tokenizer, result_txt)
        logging.info(f'test: bleu1, bleu2, bleu3, bleu4, dist1, dist2, rouge1, rouge2, rougeL, meteor, nist1, nist2, nist3, nist4: ' +
                    f'{bleu1:.4f}, {bleu2:.4f}, {bleu3:.4f}, {bleu4:.4f}, {dist1:.4f}, {dist2:.4f}, {rouge1:.4f}, {rouge2:.4f}, {rougeL:.4f}, {meteor:.4f}, {nist1:.4f}, {nist2:.4f}, {nist3:.4f}, {nist4:.4f}')
        
    logging.info('finish')

    output_file = os.path.join(output_dir, f'{args.lr}_{args.seed}.json')
    temp = {'bleu1': bleu1, 'bleu2':bleu2, 'bleu3': bleu3, 'bleu4':bleu4, 'dist1': dist1, 'dist2':dist2, 
                    'rouge1':rouge1, 'rouge2': rouge2, 'rougeL':rougeL, 'meteor':meteor, 'nist1':nist1, 'nist2':nist2, 'nist3':nist1, 'nist4': nist4}
    res.update(temp)

    with open(output_file, 'w') as f:
        json.dump(res, f)

    output_file = os.path.join(output_dir, f'{args.lr}_{args.seed}.csv')
    with open(output_file, 'w') as f:
        for val in bleu1, bleu2, bleu3, bleu4, dist1, dist2, rouge1, rouge2, rougeL, meteor, nist1, nist2, nist3, nist4:
            f.write(f'{val:.4f},')


if __name__ == '__main__':
    # torch.multiprocessing.set_start_mode('spawn')
    # nltk.download('wordnet')
    main()
