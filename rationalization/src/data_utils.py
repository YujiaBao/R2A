import numpy as np
import re
import torch
import pickle
import itertools
import collections
import random
import json
import os
import csv
import gzip
from torchtext.vocab import Vocab
from colored import fg, attr, bg
import textwrap
import spacy

def load_data(path, regression):
    '''
    path: path to the tsv file which contains just label and text
    '''
    data_len = []
    label = []

    with open(path, 'r') as f:
        next(f)
        data = []

        for line in f:
            line = line.strip().split('\t')

            data.append({
                'label': float(line[1]) if regression else int(float(line[1])), 
                'text': line[2].split()
                })

            label.append(data[-1]['label'])
            data_len.append(len(data[-1]['text']))

        counter = collections.Counter(data_len)
        print('Average length: ' + str(sum(data_len)/len(data_len)))
        label = np.array(label)
        print(collections.Counter(label))
        print("Label mean: %.4f, variance: %.4f" % (np.mean(label), np.var(label)))
        
    return data

def load_dataset_beer(args):
    aspect = int(args.dataset[-1])

    if args.mode == 'train':
        train_data = load_data('../data/source/beer%d.train' % aspect, regression=True)
        dev_data   = load_data('../data/source/beer%d.dev' % aspect, regression=True)
        test_data  = []

    elif args.mode == 'test':
        train_data = []
        dev_data   = []
        test_data  = load_data(args.test_path, regression=True)

    else:
        raise ValueError('Mode can only be train, test')

    return train_data, dev_data, test_data

def read_words(data):
    words = []
    for example in data:
        words += example['text']
    return words

def data_to_nparray(data, vocab, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    # process the text
    text_len = np.array([len(e['text']) for e in data])
    max_text_len = max(text_len)

    text = np.ones([len(data), max_text_len], dtype=np.int64) * vocab.stoi['<pad>']
    for i in range(len(data)):
        text[i,:len(data[i]['text'])] = [
                vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>'] for x in data[i]['text']]

    if args.num_classes == 1: # regression task
        doc_label = np.array([x['label'] for x in data], dtype=np.float32)

    else: # classification task
        doc_label = np.array([x['label'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)

    return {'text':      text, 
            'text_len':  text_len, 
            'label':     doc_label, 
            'raw':       raw,
            }

def load_dataset(args, vocab=None):
    print("Loading data...")
    if args.dataset[:-1] == 'beer':
        train_data, dev_data, test_data = load_dataset_beer(args)
    else:
        raise ValueError('Invalid dataset name')

    if vocab is None:
        if args.word_vector in ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d',
                'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d',
                'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d',
                'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']:
            vocab = Vocab(collections.Counter(read_words(train_data)), vectors=args.word_vector)
        else: # load pre-defined word vector
            v = torchtext.vocab.Vectors(args.word_vector)
            vocab = Vocab(collections.Counter(read_words(train_data)), vectors=v)
    
    wv_size = vocab.vectors.size()
    print('Total num. of words: %d\nWord vector dimension: %d' % (wv_size[0], wv_size[1]))

    num_oov = wv_size[0] - torch.nonzero(torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
    print('Num. of out-of-vocabulary words (they are initialized to zero vector): %d'\
            % num_oov)

    print('Length of data: train: %d, dev: %d, test: %d.' % (len(train_data), len(dev_data), len(test_data)))

    if args.mode == 'train':
        train_data = data_to_nparray(train_data, vocab, args)
        dev_data   = data_to_nparray(dev_data, vocab, args) 
    elif args.mode == 'test':
        test_data  = data_to_nparray(test_data, vocab, args)
    else:
        raise ValueError('Invalid mode')

    return train_data, dev_data, test_data, vocab

def data_loader(data, batch_size, shuffle=True):
    """
        Generates a batch iterator for a dataset.
    """
    data_size = len(data['label'])
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1

    np.random.seed(1)

    # shuffle the dataset at the begging of each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        for key, value in data.items():
            data[key] = value[shuffle_indices]

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        max_text_len = max(data['text_len'][start_index:end_index])

        yield (data['text'][start_index:end_index,:max_text_len],
                data['text_len'][start_index:end_index],
                data['label'][start_index:end_index], 
                data['raw'][start_index:end_index])

def write_human(writer, raw, pred, true, pred_rationale, show_weights=False):
    '''
    Write the prediction result (with generated rationale) to a tsv file.
    '''
    for i in range(len(raw)):
        writer.write("%.2f %.2f " % (true[i], pred[i]))

        if show_weights:
            text = (" ".join([x if len(x) > 4 else "#"*(4-len(x)) + x for x in raw[i]])).strip()
        else:
            text = (" ".join(raw[i])).strip()

        lines = textwrap.wrap(text, width=80, break_long_words=False, break_on_hyphens=False)

        idx = 0
        for j, line in enumerate(lines):
            if j != 0:
                weights = " " * 10
                colored_line = " " * 10
            else:
                weights = ""
                colored_line = ""

            for word in line.strip().split():
                color = ""  # highlight true rationale, use red font for generated rationale
                if pred_rationale[i][idx] > 0.5:
                    color += fg(1)

                colored_line += color + word.replace('#', ' ') + ' ' + attr(0)
                weights += (len(word)-4) * ' ' + ('%.2f' % pred_rationale[i][idx]) + ' '
                idx += 1

            if show_weights:
                writer.write(weights + '\n')

            writer.write(colored_line + '\n')
            writer.write('\n')

def write_machine(writer, task, raw, true, pred_rationale):
    '''
    Write the prediction result (with generated rationale) to a tsv file.
    '''
    for i in range(len(raw)):
        writer.write(task + '\t')
        writer.write("%.2f\t" % true[i])
        writer.write(" ".join(raw[i]) + '\t')
        writer.write(" ".join(["%d" % round(x) for x in pred_rationale[i][:len(raw[i])] ]) + '\n')

def filter_rationale(raw, pred, true, pred_rationale):
    new_raw = []
    new_pred = []
    new_true = []
    new_rat = []
    nlp = spacy.load('en')
    nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)

    for i in range(len(raw)):
        # if selected more than 60%, skip
        cur_rat = pred_rationale[i,:len(raw[i])]
        if sum(cur_rat)/len(cur_rat) >= 0.6:
            continue

        # if variation probability greater than 50%, skip
        if sum(np.absolute(cur_rat[1:] - cur_rat[:-1]))/len(cur_rat) >= 0.5:
            continue

        cur_rat = np.around(cur_rat)
        if sum(cur_rat) < 1:
            continue

        # if 1 0 1, then make it 1 1 1
        for j in range(1, len(cur_rat)-1):
            if cur_rat[j-1] > 0.5 and cur_rat[j+1] > 0.5:
                cur_rat[j] = 1

        # if 0 1 0, then make it 0 0 0
        for j in range(1, len(cur_rat)-1):
            if cur_rat[j-1] < 0.5 and cur_rat[j+1] < 0.5:
                cur_rat[j] = 0

        if sum(cur_rat) < 1:
            continue

        idx = 0
        doc = nlp(" ".join(raw[i]))
        for sent in doc.sents:
            sent_rat_cnt = 0.0 + sum(cur_rat[idx:idx+len(sent)])
            if sent_rat_cnt / len(sent) > 0.8:
                for j in range(idx, idx+len(sent)):
                    cur_rat[j] = 1
            else:
                for j, token in enumerate(sent):
                    if cur_rat[j+idx] == 1:
                        # punct cannot be the beginning of the rationale
                        if token.pos_ == 'PUNCT':
                            if j+idx-1 >= 0 and cur_rat[j+idx-1] != 1:
                                cur_rat[j+idx] = 0
                                continue

                        # if an ADJ is selected as rationale, the verb and noun before it should be rat
                        if token.pos_ == 'ADJ':
                            for k in range(j):
                                if sent[j-k-1].pos_ in ['NOUN', 'ADV', 'DET', 'VERB', 'ADP']:
                                    cur_rat[j+idx-k-1] = 1
                                else:
                                    break

                        # if one token is selected as rationale, all its children should be a rationale
                        for child in token.children:
                            cur_rat[child.i] = 1

                flag = True
                for j in range(len(sent)-1,-1,-1):
                    if cur_rat[j+idx] == 1:
                        if sent[j].pos_ not in ['NOUN', 'DET', 'ADP', 'PUNCT']:
                            flag = False
                            break
                    else:
                        break
                if flag:
                    for k in range(len(sent)-1,j,-1):
                        cur_rat[k+idx] = 0

                flag = True
                for j in range(len(sent)):
                    if cur_rat[j+idx] == 1:
                        if not sent[j].is_stop and \
                            sent[j].pos_ not in ['NOUN', 'DET', 'PRON', 'PUNCT', 'ADV']:
                            flag = False
                            break
                    else:
                        break

                if flag:
                    for k in range(j):
                        cur_rat[k+idx] = 0


            sent_rat_cnt = 0.0 + sum(cur_rat[idx:idx+len(sent)])
            if sent_rat_cnt / len(sent) > 0.8:
                for j in range(idx, idx+len(sent)):
                    cur_rat[j] = 1

            elif sent_rat_cnt < 2:
                for j in range(idx, idx+len(sent)):
                    cur_rat[j] = 0

            idx += len(sent)

        if sum(cur_rat) < 1:
            continue

        # if 1 0 1, then make it 1 1 1
        for j in range(1, len(cur_rat)-1):
            if cur_rat[j-1] > 0.5 and cur_rat[j+1] > 0.5:
                cur_rat[j] = 1

        new_raw.append(raw[i])
        new_pred.append(pred[i])
        new_true.append(true[i])
        new_rat.append(cur_rat)

    new_raw = np.array(new_raw, dtype=object)
    new_pred = np.array(new_pred, dtype=object)
    new_true = np.array(new_true, dtype=object)
    new_rat = np.array(new_rat, dtype=object)

    return new_raw, new_pred, new_true, new_rat

def generate_writer(path, refilter=False):
    writer = {}

    if not refilter:
        writer['human'] = open(path + '.human_readable.tsv', 'w')
        writer['machine'] = open(path + '.machine_readable.tsv', 'w')
        writer['filtered_human'] = open(path + '.human_readable.filtered.tsv', 'w')
        writer['filtered_machine'] = open(path + '.machine_readable.filtered.tsv', 'w')

        writer['human'].write("task\ttrue_lbl\tpred_lbl\ttext\n")
        writer['machine'].write("task\tlabel\ttext\trationale\n")
        writer['filtered_human'].write("task\ttrue_lbl\tpred_lbl\ttext\n")
        writer['filtered_machine'].write("task\tlabel\ttext\trationale\n")

    return writer

def close_writer(writer):
    for key in writer.keys():
        writer[key].close()
