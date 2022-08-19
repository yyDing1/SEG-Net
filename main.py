import argparse, os, random, re
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from Selector import *
from ExtractorGenerator import *


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vocab(src_file, trg_file):
    token_freq_counter = Counter()
    for src_line, trg_line in zip(open(src_file, 'r'), open(trg_file, 'r')):
        token_freq_counter.update(src_line.strip().split(' '))
        token_freq_counter.update(trg_line.strip().split(';'))

    special_tokens = ['<pad>', '<unk>', '<sep>', '<eos>', '<begin>']
    for special_token in special_tokens:
        del token_freq_counter[special_token]
    
    sorted_words2idx = sorted(token_freq_counter.items(), key=lambda x: x[1], reverse=True)
    sorted_words = [x[0] for x in sorted_words2idx]
    word2idx, idx2word = {}, {}
    for idx, word in enumerate(special_tokens):
        word2idx[word] = idx
        idx2word[idx] = word
    for idx, word in enumerate(sorted_words):
        word2idx[word] = idx + len(special_tokens)
        idx2word[idx + len(special_tokens)] = word
    return word2idx, idx2word, token_freq_counter


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=None)
parser.add_argument('--train_data_dir', type=str, default='./data/kp20k_separated')
parser.add_argument('--test_data_dir', type=str, default='./data/testsets/kp20k')
parser.add_argument('--vocab_size', type=int, default=50000)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--selector_layers', type=int, default=6)
parser.add_argument('--extractor_generator_layers', type=int, default=6)
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--dropout_rate', type=int, default=0.2)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--learning_rate_decay', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--selector_w', type=float, default=0.7)
parser.add_argument('--extractor_generator_w', type=float, default=2.0)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--selector_train_epochs', type=int, default=15)
parser.add_argument('--extractor_generator_train_epochs', type=int, default=25)
parser.add_argument('--save_path', type=str, default='.')

parser.add_argument('--build_vocab', action='store_true')
parser.add_argument('--build_selector_data', action='store_true')
parser.add_argument('--train_selector', action='store_true')

parser.add_argument('--train_extractor_generator', action='store_true')
parser.add_argument('--build_extractor_generator_data', action='store_true')


args = parser.parse_args()
if args.gpu_id:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(args.seed)

if args.build_vocab:
    word2idx, idx2word, token_freq_counter = build_vocab(
        src_file=os.path.join(args.train_data_dir, 'train_src.txt'), 
        trg_file=os.path.join(args.train_data_dir, 'train_trg.txt')
        )
    torch.save([word2idx, idx2word, token_freq_counter], open(os.path.join(args.save_path, 'vocab.pt'), 'wb'))

if args.build_selector_data:
    word2idx, idx2word, token_freq_counter = torch.load(os.path.join(args.save_path, 'vocab.pt'), 'wb')
    train_data = build_selector_data(
        src_file=os.path.join(args.train_data_dir, 'train_src.txt'), 
        trg_file=os.path.join(args.train_data_dir, 'train_trg.txt'),
        word2idx=word2idx,
        vocab_size=args.vocab_size
        )
    valid_data = build_selector_data(
        src_file=os.path.join(args.train_data_dir, 'valid_src.txt'), 
        trg_file=os.path.join(args.train_data_dir, 'valid_trg.txt'),
        word2idx=word2idx,
        vocab_size=args.vocab_size
        )
    selector_train_dataset = SelectorDataset(train_data)
    selector_valid_dataset = SelectorDataset(valid_data)
    torch.save({
        'train_dataset': selector_train_dataset,
        'valid_dataset': selector_valid_dataset
    }, open(os.path.join(args.save_path, 'selector_dataset.pt'), 'wb'))

if args.train_selector:
    train_selector(args)

if args.build_extractor_generator_data:
    word2idx, idx2word, token_freq_counter = torch.load(os.path.join(args.save_path, 'vocab.pt'), 'wb')
    train_data = build_extractor_generator_data(
        src_file=os.path.join(args.train_data_dir, 'train_src.txt'), 
        trg_file=os.path.join(args.train_data_dir, 'train_trg.txt'),
        word2idx=word2idx,
        vocab_size=args.vocab_size
        )
    valid_data = build_extractor_generator_data(
        src_file=os.path.join(args.train_data_dir, 'valid_src.txt'), 
        trg_file=os.path.join(args.train_data_dir, 'valid_trg.txt'),
        word2idx=word2idx,
        vocab_size=args.vocab_size
        )
    extractor_generator_train_dataset = ExtractorGeneratorDataset(train_data)
    extractor_generator_valid_dataset = ExtractorGeneratorDataset(valid_data)
    torch.save({
        'train_dataset': extractor_generator_train_dataset,
        'valid_dataset': extractor_generator_valid_dataset
    }, open(os.path.join(args.save_path, 'extractor_generator_dataset.pt'), 'wb'))
    
if args.train_extractor_generator:
    train_extractor_generator(args)

