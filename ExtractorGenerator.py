import re, os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from MyTransformers import ExtractorGenerator, ExtractorGeneratorLoss
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence



def build_extractor_generator_data(src_file, trg_file, word2idx, vocab_size):
    src_input_list, tgt_input_list, extractor_labels, generator_labels, copy_mask_list, segment_id_list = [], [], [], [], [], []
    for src_line, trg_line in zip(open(src_file, 'r'), open(trg_file, 'r')):
        if len(src_line.strip()) == 0:
            continue
        sentence_list = []
        pattern = r',|\.'
        for sentence in re.split(pattern, src_line.strip()):
            if len(sentence.strip()) > 0:
                sentence_list.append(sentence.strip())
        kg_all = trg_line.strip().split(';')
        if '<peos>' not in kg_all:
            raise Exception('Error, no <peos> in keyword phrases!')
        present, absent = kg_all[:kg_all.index('<peos>')], kg_all[kg_all.index('<peos>') + 1:]

        present_words = []
        for keyphrase in present:
            present_words.extend(keyphrase.strip().split())

        absent_words = []
        for keyphrase in absent:
            absent_words.extend(keyphrase.strip().split())
            absent_words.extend(['<sep>'])
        absent_words.extend(['<eos>'])

        all_match_parts = present + absent_words

        salient_sentence_list, segment_id = [], []
        for idx, sentence in enumerate(sentence_list):
            for elem in all_match_parts:
                if elem in sentence:
                    salient_sentence_list.extend(sentence.strip().split())
                    segment_id.extend([idx] * len(sentence.strip().split()))
                    break

        extractor_target = [0] * len(salient_sentence_list)
        for elem in all_match_parts:
            now_part = elem.split()
            for idx in range(len(salient_sentence_list) - len(now_part) + 1):
                if salient_sentence_list[idx: idx + len(now_part)] == now_part:
                    extractor_target[idx: idx + len(now_part)] = [1] * len(now_part)
        
        copy_mask = []
        for tgt_word in absent_words:
            copy_mask.append([])
            for src_word in salient_sentence_list:
                copy_mask[-1].append(1 if tgt_word == src_word else 0)
        
        salient_sentence_list = [word2idx[w] if w in word2idx and word2idx[w] < vocab_size else word2idx['<unk>'] for w in salient_sentence_list]
        absent_words = [word2idx[w] if w in word2idx and word2idx[w] < vocab_size else word2idx['<unk>'] for w in absent_words]

        if len(salient_sentence_list) == 0 or len(absent_words) == 0:
            continue
        
        src_input_list.append(salient_sentence_list)
        tgt_input_list.append([4] + absent_words[:-1])  # begin token
        extractor_labels.append(extractor_target)
        generator_labels.append(absent_words)
        copy_mask_list.append(copy_mask)
        segment_id_list.append(segment_id)

    return {
        'src_input': src_input_list,
        'tgt_input': tgt_input_list,
        'extractor_labels': extractor_labels,
        'generator_labels': generator_labels,
        'copy_mask': copy_mask_list,
        'segment_id': segment_id_list
    }


def extractor_generator_metric():
    pass


class ExtractorGeneratorDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.src_input = data['src_input']
        self.tgt_input = data['tgt_input']
        self.extractor_labels = data['extractor_labels']
        self.generator_labels = data['generator_labels']
        self.copy_mask = data['copy_mask']
        self.segment_id = data['segment_id']

    def __len__(self):
        return len(self.src_input)
    
    def __getitem__(self, index):
        return torch.LongTensor(self.src_input[index]), torch.LongTensor(self.tgt_input[index]), torch.LongTensor(self.extractor_labels[index]), \
                torch.LongTensor(self.generator_labels[index]), torch.Tensor(self.copy_mask[index]), torch.LongTensor(self.segment_id[index]), index


def extractor_generator_train_epoch(model, dataloader, optimizer, loss_func, epoch_id, device):
    model.train()
    pbar = tqdm(dataloader)
    pbar.set_description('Training Epoch {}'.format(epoch_id))
    for src_input, tgt_input, extractor_target, generator_target, copy_masks, seg_ment_ids, index in dataloader:
        src_input, tgt_input, extractor_target, generator_target, copy_masks, seg_ment_ids, index = \
            src_input.to(device), tgt_input.to(device), extractor_target.to(device), generator_target.to(device), copy_masks.to(device), seg_ment_ids.to(device), index.to(device)
        model.zero_grad()
        extractor_output, generator_output, copy_output = model(src_input, tgt_input, copy_masks, seg_ment_ids)
        loss = loss_func(extractor_output, generator_output, copy_output, extractor_target, generator_target, src_input > 0, tgt_input > 0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        pbar.set_postfix(loss=loss.item())


    
def train_extractor_generator(args):
    def collate_fn(data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        src_input = pad_sequence([x[0] for x in data], batch_first=True, padding_value=0)
        tgt_input = pad_sequence([x[1] for x in data], batch_first=True, padding_value=0)
        extractor_labels = pad_sequence([x[2] for x in data], batch_first=True, padding_value=0)
        generator_labels = pad_sequence([x[3] for x in data], batch_first=True, padding_value=0)
        mask_dim1, mask_dim2 = max([len(x[4]) for x in data]), max([len(x[4][0]) for x in data])
        copy_masks = torch.zeros(len(data), mask_dim1, mask_dim2, dtype=int)
        for idx, x in enumerate(data):
            copy_masks[idx, :len(x[4]), :len(x[4][0])] = torch.Tensor(x[4])
        segment_ids = pad_sequence([x[5] for x in data], batch_first=True, padding_value=0)
        index = torch.LongTensor([x[6] for x in data])
        return src_input, tgt_input, extractor_labels, generator_labels, copy_masks, segment_ids, index

    extractor_generator_dataset = torch.load(os.path.join(args.save_path, 'extractor_generator_dataset.pt'), 'wb')
    extractor_generator_train_dataset, extractor_generator_valid_dataset = extractor_generator_dataset['train_dataset'], extractor_generator_dataset['valid_dataset']
    extractor_generator_train_dataloader = DataLoader(dataset=extractor_generator_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    extractor_generator_valid_dataloader = DataLoader(dataset=extractor_generator_valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print('Train data length = {}, Valid data length = {}'.format(len(extractor_generator_train_dataset), len(extractor_generator_valid_dataset)))
    extractor_generator = ExtractorGenerator(args.d_model, args.num_head, args.d_ff, args.dropout_rate, args.extractor_generator_layers, args.vocab_size).to(args.device)
    extractor_generator.init_params()

    optimizer = torch.optim.Adam(extractor_generator.parameters(), args.learning_rate)
    loss_func = ExtractorGeneratorLoss(args.extractor_generator_w, args.beta, args.vocab_size)

    best_f1, half = 0, False
    for epoch in range(1, args.extractor_generator_train_epochs + 1):
        extractor_generator_train_epoch(extractor_generator, extractor_generator_train_dataloader, optimizer, loss_func, epoch, args.device)
    #     train_f1, _ = extractor_generator_evaluate_epoch(extractor_generator, extractor_generator_train_dataloader, args.device, 'train')
    #     valid_f1, _ = extractor_generator_evaluate_epoch(extractor_generator, extractor_generator_valid_dataloader, args.device, 'valid')
    #     if valid_f1 >= best_f1:
    #         best_f1 = valid_f1
    #         torch.save({
    #             'epoch': epoch,
    #             'train_f1': train_f1,
    #             'valid_f1': valid_f1,
    #             'model_state_dict': extractor_generator.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #         }, os.path.join(args.save_path, 'extractor_generator.pt'))
    #     elif not half:
    #         print('Valid f1 drops, half the learning rate ...')
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] *= 0.5
    #         half = True