import numpy as np
import torch
import math
import pickle
import itertools
import collections
import random
import json
import os
import csv
import gzip
import torchtext
from torchtext.vocab import Vocab

def load_rationale(path, regression):
    '''
    path: path to the tsv file that has machine generated rationales
    regression: a binary variable indicating whether we are doing classification or regression
    '''
    label = []
    with open(path, 'r') as f:
        next(f)
        data = []

        for row in f:
            row = row.strip().split('\t')
            data.append({
                'label'    : float(row[1]) if regression else int(float(row[1])),
                'text'     : row[2].strip().split(' '), 
                'rationale': [int(x) for x in row[3].strip().split(' ')],
                'task'     : row[0].strip(),
                })
            label.append(data[-1]['label'])

    label = np.array(label)
    counter = sorted(collections.Counter(label).items())
    print('Label distribution:', end=' ')
    print(counter)

    if regression:
        print('Label mean: %.4f, var: %.4f' % (np.mean(label), np.var(label)))

    return data

def load_rat_pred_gold(path, task, regression):
    '''
    path: path to the tsv file which contains label, text, oracle attention, rationale and R2A
    generated attention
    task: the name of the task
    regression: a binary variable indicating whether we are doing classification or regression
    '''
    with open(path, 'r') as f:
        next(f)
        data = []

        for row in f:
            row = row.strip().split('\t')

            # Only collect examples for this aspect
            if task != row[0].strip():
                continue

            text = row[2].replace('  ',' ').split(' ')
            rationale = [0 for i in range(len(text))]
            pred_att  = [0 for i in range(len(text))]
            gold_att  = [0 for i in range(len(text))]

            if len(row) > 3:
                rationale = [round(float(x)) for x in row[3].strip().split(' ')]
            if len(row) > 4:
                pred_att = [float(x) for x in row[4].strip().split(' ')]
            if len(row) > 5:
                gold_att = [float(x) for x in row[5].strip().split(' ')]

            data.append({
                'text'     : text, 
                'label'    : float(row[1]) if regression else int(float(row[1])),
                'rationale': rationale,
                'gold_att' : gold_att,
                'pred_att' : pred_att,
                })

    return data

def load_data(path):
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

            data.append({'label': int(line[1]), 'text': line[2].split()})

            label.append(data[-1]['label'])
            data_len.append(len(data[-1]['text']))

        counter = collections.Counter(data_len)
        print('Average length: ' + str(sum(data_len)/len(data_len)))
        label = np.array(label)
        print(collections.Counter(label))
        print("Label mean: %.4f, variance: %.4f" % (np.mean(label), np.var(label)))
        
    return data

def load_unlabeled(path):
    '''
    path: path to the tsv file which contains just text
    '''
    data_len = []

    with open(path, 'r') as f:
        data = []

        for line in f:
            line = line.strip()
            data.append({'text': line.split()})
            data_len.append(len(data[-1]['text']))

        counter = collections.Counter(data_len)
        print('Average length: ' + str(sum(data_len)/len(data_len)))
        
    return data

def write_dataset(path, task, data):
    with open(path, 'w') as f:
        f.write('task\tlabel\ttext\n')
        for example in data:
            f.write(task + '\t' + str(example['label']) + '\t' + " ".join(example['text']) + '\n')

def load_dataset_helper(task, num_classes, args):
    print("\nLoading data " + task)
    if task[:-1] == 'beer':
        return load_dataset_beer(task, num_classes, args)

    elif task[:5] == 'hotel':
        return load_dataset_hotel(task, num_classes, args)

    else:
        raise ValueError("Invalid task name.")

def load_dataset_beer(dataset, num_classes, args):
    datalist = { 
        'beer0': '../data/source/beer0',
        'beer1': '../data/source/beer1',
        'beer2': '../data/source/beer2',
        }# path to the source data (with machine rationales)

    if args.mode == 'train_r2a': # training R2A
        # regression task on the source
        train_data = load_rationale(datalist[dataset] + '.train', True)
        dev_data   = load_rationale(datalist[dataset] + '.dev', True)

        print("len: train %d, dev %d" % (len(train_data), len(dev_data)))

        return train_data, dev_data, None
    
    elif args.mode == 'test_r2a' or args.mode == 'test_clf': # R2A inference or classifier inference
        test_data = load_rat_pred_gold(args.test_path, dataset, True)
        print("len: test data %d" % len(test_data))

        return None, None, test_data

    elif args.mode == 'train_clf': # training the target classifier
        # classification task
        aspect = int(dataset[-1])

        if args.train_path == '':
            print("Path to the training data not specified. Loading the oracle data")
            train_data = load_data('../data/oracle/' + dataset + '.train')
            dev_data   = load_data('../data/oracle/' + dataset + '.dev')
        else:
            print("Loading the specified training data")
            train_data = load_rat_pred_gold(args.train_path, dataset, regression=False)
            dev_data   = load_data('../data/target/' + dataset + '.dev')

        test_data  = load_data('../data/target/' + dataset + '.test')
        print("len: train %d, dev %d, test %d" % (len(train_data), len(dev_data), len(test_data)))

        return train_data, dev_data, test_data

def load_dataset_hotel(dataset, num_classes, args):
    if args.mode == 'train_r2a': # preparing unlabeled data for R2A training
        train_data = load_unlabeled('../data/target/hotel_unlabeled.train')
        dev_data   = load_unlabeled('../data/target/hotel_unlabeled.dev')

        print("len: train %d, dev %d" % (len(train_data), len(dev_data)))

        return train_data, dev_data, None

    elif args.mode == 'test_r2a' or args.mode == 'test_clf': # R2A inference or classifier inference
        test_data = load_rat_pred_gold(args.test_path, dataset, True)
        print("len: test data %d" % len(test_data))

        return None, None, test_data

    elif args.mode == 'train_clf': # Training a new target classifier
        # classification task
        aspect = dataset[6:]
        if args.train_path == '':
            print("Path to the training data not specified. Loading the oracle data")
            train_data = load_data('../data/oracle/' + dataset + '.train')
            dev_data   = load_data('../data/oracle/' + dataset + '.dev')
        else:
            print("Loading the specified training data")
            train_data = load_rat_pred_gold(args.train_path, dataset, regression=False)
            dev_data   = load_data('../data/target/' + dataset + '.dev')

        test_data  = load_data('../data/target/' + dataset + '.test')
        print("len: train %d, dev %d, test %d" % (len(train_data), len(dev_data), len(test_data)))

        return train_data, dev_data, test_data

def read_words(data_dict):
    '''
    Get a list of words that appear in the data dictionary.
    '''
    words = []
    for data in data_dict.values():
        for example in data:
            words += example['text']
    return words

def data_to_nparray(data_dict, vocab, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    output = {}

    # get maximum text length
    max_text_len = 0
    for task, data in data_dict.items():
        if data is not None:
            output[task] = {}
            output[task]['text_len'] = np.array([len(e['text']) for e in data])
            max_text_len = max(max_text_len, max(output[task]['text_len']))
        else:
            output[task] = None

    for task, data in data_dict.items():
        if data is None:
            continue

        output[task]['text'] = np.ones([len(data), max_text_len], dtype=np.int64)\
                                 * vocab.stoi['<pad>']

        for i in range(len(data)):
            output[task]['text'][i,:len(data[i]['text'])] = [
                vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>'] for x in data[i]['text']]

        if args.num_classes[task] == 1: # regression task
            if 'label' in data[0]:
                output[task]['label'] = np.array([x['label'] for x in data], dtype=np.float32)
            else:
                output[task]['label'] = np.zeros(len(data), dtype=np.float32)
        else: # classification task
            if 'label' in data[0]:
                output[task]['label'] = np.array([x['label'] for x in data], dtype=np.int64)
            else:
                output[task]['label'] = np.zeros(len(data), dtype=np.int64)

        output[task]['raw'] = np.array([e['text'] for e in data], dtype=object)

        output[task]['rationale'] = np.zeros([len(data), max_text_len], dtype=np.float32)
        for i in range(len(data)):
            if 'rationale' in data[i]:
                output[task]['rationale'][i,:len(data[i]['rationale'])] = data[i]['rationale']

        output[task]['gold_att'] = np.zeros([len(data), max_text_len], dtype=np.float32)
        if 'gold_att' in data[0]:
            for i in range(len(data)):
                output[task]['gold_att'][i,:len(data[i]['gold_att'])] = data[i]['gold_att']

        output[task]['pred_att'] = np.zeros([len(data), max_text_len], dtype=np.float32)
        if 'pred_att' in data[0]:
            for i in range(len(data)):
                output[task]['pred_att'][i,:len(data[i]['pred_att'])] = data[i]['pred_att']

        # compute occurrences of each word
        word_all, word_all_cnts = np.unique(output[task]['text'], return_counts=True)
        all_dict = dict(zip(word_all, word_all_cnts))

        # compute occurrences of each rationale word
        word_rat, word_rat_cnts = np.unique(
                output[task]['text'][output[task]['rationale']>0.5], return_counts=True)
        rat_dict = dict(zip(word_rat, word_rat_cnts))

        # pre-compute the frequency of the word being a rationale in this task
        text_cnt = np.ones(output[task]['text'].shape, dtype=np.float32)
        text_rat_cnt = np.zeros(output[task]['text'].shape, dtype=np.float32)
        for i in range(len(data)):
            for j in range(len(data[i]['text'])):
                tkn = output[task]['text'][i,j]
                text_cnt[i,j] = all_dict[tkn]
                text_rat_cnt[i,j] = 0 if tkn not in rat_dict else rat_dict[tkn]
        output[task]['rat_freq'] = text_rat_cnt / text_cnt

    return output

def load_dataset(args, vocab=None):
    '''
    Load the source and target data. 
    Load the pretrained word emebeeding.
    Convert the data into np arrays for fast training.
    '''
    train_data_dict, dev_data_dict, test_data_dict = {}, {}, {}

    print("\n=== Loading Source ===")
    if args.mode != 'test_r2a':
        for task in args.src_dataset:
            train_data_dict[task], dev_data_dict[task], test_data_dict[task] = load_dataset_helper(
                    task, args.num_classes[task], args)

    print("\n=== Loading Target (The label of the target data will not be used during training r2a) ===")
    if args.tar_dataset != '':
        train_data_dict[args.tar_dataset], dev_data_dict[args.tar_dataset], test_data_dict[args.tar_dataset] \
                = load_dataset_helper(args.tar_dataset, args.num_classes[args.tar_dataset], args)

    if vocab is None:
        if args.word_vector in ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d',
                'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d',
                'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d',
                'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']:
            vocab = Vocab(collections.Counter(read_words(train_data_dict)), vectors=args.word_vector,
                    min_freq=5)
        else: # load pre-defined word vector
            v = torchtext.vocab.Vectors(args.word_vector)
            vocab = Vocab(collections.Counter(read_words(train_data_dict)), vectors=v, min_freq=5)

    wv_size = vocab.vectors.size()
    print('\nNum. of words: %d\nWord vector dimension: %d' % (wv_size[0], wv_size[1]))

    num_oov = wv_size[0] - torch.nonzero(torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
    print('Num. of OOV: %d (they are initialized to zero vector)' % num_oov)

    train_data_dict = data_to_nparray(train_data_dict, vocab, args)
    dev_data_dict   = data_to_nparray(dev_data_dict, vocab, args) 
    test_data_dict  = data_to_nparray(test_data_dict, vocab, args)

    return train_data_dict, dev_data_dict, test_data_dict, vocab

def data_loader(origin_data, batch_size, shuffle=True, oneEpoch=True):
    """
        Generates a batch iterator for a dataset.
    """
    # copy the original data
    data = {}
    for key, value in origin_data.items():
        data[key] = np.copy(value)

    data_size = len(data['label'])
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1 if oneEpoch else data_size//batch_size

    cnts = 0.0
    while True:
        # shuffle the dataset at the begging of each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            for key, value in data.items():
                data[key] = value[shuffle_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            cnts += end_index - start_index + 1

            max_text_len = max(data['text_len'][start_index:end_index])

            yield (data['text'][start_index:end_index,:max_text_len],
                    data['rat_freq'][start_index:end_index,:max_text_len],
                    data['rationale'][start_index:end_index,:max_text_len],
                    data['gold_att'][start_index:end_index,:max_text_len],
                    data['pred_att'][start_index:end_index,:max_text_len],
                    data['text_len'][start_index:end_index],
                    data['label'][start_index:end_index], 
                    data['raw'][start_index:end_index],
                    cnts/data_size)

        if oneEpoch:
            break

def data_dict_loader(data_dict, task_list, batch_size):
    '''
        Generates a batch iterator for a dataset dictionary.
        Maintain an iterator for each task
        Output a dictionary of batches, where each key correspond to each task
    '''
    loader = {}
    for task in task_list:
        loader[task] = data_loader(data_dict[task], batch_size, True, False)

    while True:
        output = {}

        for key in loader.keys():
            output[key] = next(loader[key])

        yield output

def write(writer, task, raw, label, gold_att, rationale, pred_att, rat_freq=None, att=None):
    for i in range(len(raw)):
        writer.write(task + '\t' + str(label[i]) + '\t' + \
            " ".join([x for x in raw[i]]) + '\t' + \
            " ".join(['%d' % round(x) for x in rationale[i,:len(raw[i])] ]) + '\t' + \
            " ".join(['%.8f' % x for x in pred_att[i,:len(raw[i])] ]) + '\t' + \
            " ".join(['%.8f' % x for x in gold_att[i,:len(raw[i])] ]))

        if isinstance(rat_freq, np.ndarray):
            writer.write('\t' + " ".join(['%.4f' % x for x in rat_freq[i,:len(raw[i])] ]))

        if isinstance(att, np.ndarray):
            writer.write('\t' + " ".join(['%.8f' % x for x in att[i,:len(raw[i])] ]))

        writer.write('\n')

