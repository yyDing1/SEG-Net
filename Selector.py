import re, os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from MyTransformers import Selector
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def build_selector_data(src_file, trg_file, word2idx, vocab_size):
    train_src, trg_present, trg_absent, src_hav_kg = [], [], [], []
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
        absent_words = []
        for keywords_phrase in absent:
            absent_words.extend(keywords_phrase.split())
        hav_kg = [0] * len(sentence_list)
        all_match_parts = present + absent_words
        for idx, sentence in enumerate(sentence_list):
            for elem in all_match_parts:
                if elem in sentence:
                    hav_kg[idx] = 1
                    break
        for idx, sentence in enumerate(sentence_list):
            sentence_list[idx] = [word2idx[w] if w in word2idx and word2idx[w] < vocab_size else word2idx['<unk>'] for w in sentence.split()]
        present_idx = [word2idx[w] if w in word2idx and word2idx[w] < vocab_size else word2idx['<unk>'] for w in present]
        absent_idx = [word2idx[w] if w in word2idx and word2idx[w] < vocab_size else word2idx['<unk>'] for w in absent]
        
        train_src.append(sentence_list)
        trg_present.append(present_idx)
        trg_absent.append(absent_idx)
        src_hav_kg.append(hav_kg)
    
    return {'src': train_src, 
            'src_hav_kg': src_hav_kg,
            'trg_present': trg_present, 
            'trg_absent': trg_absent}


def selector_metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    print('Accuracy: {:.1%}    Precision: {:.1%}    Recall: {:.1%}    F1: {:.1%}'
                .format(accuracy, macro_precision, macro_recall, macro_f1))
    # print(classification_report(y_true, y_pred))
    return accuracy_score, macro_precision, macro_recall, macro_f1


class SelectorDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        train_src, src_hav_kg = data['src'], data['src_hav_kg']
        self.sentence = [sentence for sentences in train_src for sentence in sentences]
        # sumlen, maxlen = 0, 0
        # for v in self.sentence:
        #     sumlen += len(v)
        #     maxlen = max(maxlen, len(v))
        # print('Avg = {}, Max = {}'.format(sumlen / len(self.sentence), maxlen))
        self.src_hav_kg = [kg for kgs in src_hav_kg for kg in kgs]
        # print(sum(self.src_hav_kg) / len(self.src_hav_kg))
        self.segment_idx = [idx for sentences in train_src for idx, sentence in enumerate(sentences)]
        assert(len(self.sentence) == len(self.src_hav_kg))
    
    def __len__(self):
        return len(self.sentence)
    
    def __getitem__(self, index):
        return torch.LongTensor(self.sentence[index]), self.src_hav_kg[index], self.segment_idx[index], index

def selector_train_epoch(model, dataloader, optimizer, loss_func, epoch_id, device):
    model.train()
    pbar = tqdm(dataloader)
    pbar.set_description('Training Epoch {}'.format(epoch_id))
    for sentence_tokens, hav_kg, segment_index, index in pbar:
        input, labels, segment_index = sentence_tokens.to(device), hav_kg.to(device), segment_index.to(device)
        model.zero_grad()
        output = model(input, segment_index)
        loss = loss_func(output, labels.float().unsqueeze(1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        pbar.set_postfix(loss=loss.item())


def selector_evaluate_epoch(model, dataloader, device, prefix):
    model.eval()
    pbar = tqdm(dataloader)
    data_num = len(dataloader.dataset)
    y_pred, y_true = np.zeros(data_num), np.zeros(data_num)
    pbar.set_description('[Evaluating on {} set]'.format(prefix))
    with torch.no_grad():
        for sentence_tokens, hav_kg, segment_index, index in pbar:
            input, labels, segment_index = sentence_tokens.to(device), hav_kg.to(device), segment_index.to(device)
            model.zero_grad()
            y_pred[index] = np.round(torch.sigmoid(model(input, segment_index).squeeze(1)).detach().cpu().numpy())
            y_true[index] = labels.detach().cpu().numpy()
    _, _, _, macro_f1 = selector_metric(y_true, y_pred)
    return macro_f1, y_pred


def train_selector(args):
    def collate_fn(data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        sentence_tokens = pad_sequence([x[0] for x in data], batch_first=True, padding_value=0)
        hav_kg = torch.LongTensor([x[1] for x in data])
        segment_index = torch.LongTensor([x[2] for x in data])
        index = torch.LongTensor([x[3] for x in data])
        return sentence_tokens, hav_kg, segment_index, index
    
    selector_dataset = torch.load(os.path.join(args.save_path, 'selector_dataset.pt'), 'wb')
    selector_train_dataset, selector_valid_dataset = selector_dataset['train_dataset'], selector_dataset['valid_dataset']
    selector_train_dataloader = DataLoader(dataset=selector_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    selector_valid_dataloader = DataLoader(dataset=selector_valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    print('Train data length = {}, Valid data length = {}'.format(len(selector_train_dataset), len(selector_valid_dataset)))
    selector = Selector(args.d_model, args.num_head, args.d_ff, args.dropout_rate, args.selector_layers, args.vocab_size).to(args.device)
    selector.init_params()

    optimizer = torch.optim.Adam(selector.parameters(), args.learning_rate)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.selector_w))
    best_f1, half = 0, False
    for epoch in range(1, args.selector_train_epochs + 1):
        selector_train_epoch(selector, selector_train_dataloader, optimizer, loss_func, epoch, args.device)
        train_f1, _ = selector_evaluate_epoch(selector, selector_train_dataloader, args.device, 'train')
        valid_f1, _ = selector_evaluate_epoch(selector, selector_valid_dataloader, args.device, 'valid')
        if valid_f1 >= best_f1:
            best_f1 = valid_f1
            torch.save({
                'epoch': epoch,
                'train_f1': train_f1,
                'valid_f1': valid_f1,
                'model_state_dict': selector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_path, 'selector.pt'))
        elif not half:
            print('Valid f1 drops, half the learning rate ...')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            half = True
