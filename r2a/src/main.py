import argparse
import sys
from os.path import dirname, realpath
import data_utils as data_utils
import numpy as np
import model_utils as model_utils
import train_utils as train_utils
import os
import torch
import datetime
import pickle

parser = argparse.ArgumentParser(description='R2A')

# ------------------------------------------------------------------------
# Training Settings
# ------------------------------------------------------------------------
parser.add_argument('--dropout',    type=float, default=0.1, 
    help='Dropout zero probability [default: 0.1]')
parser.add_argument('--lr',         type=float, default=0.001, 
    help='initial learning rate [default: 0.001]')
parser.add_argument('--batch_size', type=int, default=50,
    help='batch size for training [default: 50]')
parser.add_argument('--epoch_size', type=int, default=100,
    help='number of batches per evaluation on the dev (used only for R2A training) [default: 100]')
parser.add_argument('--patience',   type=int, default=5, 
    help='Divide the current learning rate by 10 when dev loss stop improving during the last '
        '"patience" evaluations. [default: 5]')
parser.add_argument('--fine_tune_encoder', action='store_true', default=False)
# attention supervision 
parser.add_argument('--att_target', type=str, default='pred_att', 
    help='Options: rationale, pred_att, gold_att. [default: pred_att]')
parser.add_argument('--l_r2a', type=float, default=1,
    help='Weight of the distance between R2A-generated attention and the gold attention')
parser.add_argument('--l_a2r', type=float, default=0,
    help='Weight of the distance between R2A-generated attention and the input rationale')
# Wasserstein distance
parser.add_argument('--l_wd', type=float, default=1,
    help='Weight of the Wasserstein distance')
parser.add_argument('--l_grad_penalty', type=float, default=10,
    help='Weight of the gradient penalty for the critic network')
parser.add_argument('--critic_steps', type=int, default=5, 
    help='Training steps for the critic network')
# language model
parser.add_argument('--l_lm', type=float, default=0,
    help='Language model reconstruction loss')
parser.add_argument('--lm_bins', type=int, default=100,
    help='Language model bins')

# ------------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------------
# embedding layer
parser.add_argument('--word_vector', type=str, default='fasttext.en.300d', 
    help='Name of pretrained word embeddings. Options: charngram.100d fasttext.en.300d ' \
    'fasttext.simple.300d glove.42B.300d glove.840B.300d glove.twitter.27B.25d ' \
    'glove.twitter.27B.50d glove.twitter.27B.100d glove.twitter.27B.200d glove.6B.50d ' \
    'glove.6B.100d glove.6B.200d glove.6B.300d [Default: fasttext.simple.300d]')
parser.add_argument('--fine_tune_wv',type=int, default=0,
    help='Set this to > 0 to fine tune word vectors.')
# classifier, attention, RNN
parser.add_argument('--hidden_dim',  type=int, default=50, 
    help='Dim. of the hidden layer (between conv and output) [default: 50]')
parser.add_argument('--proj_dim',    type=int, default=50, 
    help='Dim. of the projection layer in the attention module [default: 50]')
parser.add_argument('--num_layers',  type=int,   default=1,
    help='Num. of stacked RNNs [default:1]')
parser.add_argument('--cell_type',   type=str, default="LSTM",
    help="cell type: LSTM, GRU")
parser.add_argument('--rnn_size',    type=int, default=200, 
    help='Dim. of the hidden state for the RNN [default: 200]')

# ------------------------------------------------------------------------
# data & task specification
# ------------------------------------------------------------------------
parser.add_argument('--mode', type=str, default='train_r2a',
    help='Mode of the model. Options: train_r2a (training R2A), test_r2a (inference R2A), train_clf'
    '(train a new target classifier), test_clf (test the new classifier). [Default: train_r2a]')
parser.add_argument('--train_path',   type=str, default='',
    help='Path to the training data for mode train_clf. [Default: \'\']')
parser.add_argument('--test_path',   type=str, default='',
        help='Path to the testing data for mode test_clf and test_r2a. [Default: \'\']' )

parser.add_argument('--num_classes', type=str, default="2",
        help='List of number of classes for source and target tasks (separated by comma). [Default:\
        2]')
parser.add_argument('--src_dataset',  type=str, default='',
    help='Dataset used to train the r2a model for mode=train, dataset used to train the target'
    'classifier for mode=train_supervision')
parser.add_argument('--tar_dataset',  type=str, default='',
        help='Dataset used to align the r2a model [default: \'\']')

# ------------------------------------------------------------------------
# Other options
# ------------------------------------------------------------------------
parser.add_argument('--cuda',      action='store_true', default=False, help='run on gpu')
parser.add_argument('--save',      action='store_true', default=False, help='save model snapshot after training')
parser.add_argument('--dispatcher',action='store_true', default=False, help='Run from dispatcher or not')
parser.add_argument('--snapshot',    type=str, default=None,
    help='path for loading model snapshot [default: None]')
parser.add_argument('--result_path', type=str, default=None, 
    help='Path to store a pickle file of the resulting performance [default: None]')
parser.add_argument('--torch_seed', type=int, default=226,
    help='Path to store a pickle file of the resulting performance [default: 226 (bday)]')



args = parser.parse_args()
args.src_dataset = [] if args.src_dataset == '' else [dataset for dataset in args.src_dataset.split(',')]
args.num_classes = [int(num_classes) for num_classes in args.num_classes.split(',')]
args.num_classes = dict(zip(args.src_dataset + [args.tar_dataset], args.num_classes))

torch.manual_seed(args.torch_seed)
np.random.seed(1)

if __name__ == '__main__':
    # print arguments
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # load vocabulary
    if args.snapshot is not None:
        vocab = pickle.load(open(args.snapshot + '.vocab', 'rb'))
    else:
        vocab = None

    # load data
    train_data_dict, dev_data_dict, test_data_dict, vocab = data_utils.load_dataset(args, vocab)

    # Load model
    model = model_utils.get_model(vocab, args)

    if args.mode == 'train_r2a':
        '''
        Training R2A on labeled source and unlabeled target
        '''
        dev_res, saved_path, model = train_utils.train(train_data_dict, dev_data_dict, model, args)

        # saving the vocabulary
        if args.save:
            with open(saved_path+'.vocab', 'wb') as f:
                pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

        # evaluate performance on the source train & dev set
        tar_train = None if args.tar_dataset == '' else train_data_dict[args.tar_dataset]
        tar_dev   = None if args.tar_dataset == '' else dev_data_dict[args.tar_dataset]

        print("\n=== train ====")
        train_res = []
        for task in args.src_dataset:
            cur_res = train_utils.evaluate_task(
                    train_data_dict[task], task, tar_train, model, None, args)
            train_res.append(cur_res)
        train_res = train_utils.print_dev_res(train_res, args)

        print("\n=== Dev ====")
        dev_res = []
        for task in args.src_dataset:
            cur_res = train_utils.evaluate_task(
                    dev_data_dict[task], task, tar_dev, model, None, args)
            dev_res.append(cur_res)
        dev_res = train_utils.print_dev_res(dev_res, args)

    elif args.mode == 'test_r2a':
        '''
        Inference of R2A. Write the R2A-generated attention to a tsv file.
        '''
        _, _, test_data, _ = data_utils.load_dataset(args, vocab)
        writer = open(args.test_path.replace('.train', '.pred_att.train'), 'w')
        train_utils.evaluate_r2a(test_data[args.tar_dataset], args.tar_dataset, model, args, writer)
        writer.close()

    elif args.mode == 'train_clf':
        '''
        Train a classifier.
        '''
        dev_res, saved_path, model = train_utils.train(train_data_dict, dev_data_dict, model, args)

        # saving the vocabulary
        if args.save:
            with open(saved_path+'.vocab', 'wb') as f:
                pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

        # evaluate performance on the source train & dev set
        task = args.src_dataset[0]
        print("\n=== train ====")
        train_res = train_utils.evaluate_task(
                    train_data_dict[task], task, None, model, None, args)

        print("\n=== Dev ====")
        dev_res = train_utils.evaluate_task(
                    dev_data_dict[task], task, None, model, None, args)

        print("\n=== Test ====")
        test_res = train_utils.evaluate_task(
                    test_data_dict[task], task, None, model, None, args)

    else:
        '''
        Test a pre-trained classifier and write the attention to the tsv file.
        This is used for getting the oracle attention.
        '''
        print("\n=== Test ====")
        task = args.src_dataset[0]
        if args.test_path == '':
            writer = None
        else:
            writer = open(args.test_path.replace('.train', '.gold_att.train'), 'w')

        test_res = train_utils.evaluate_task(
                    test_data_dict[task], task, None, model, None, args, writer)

        if args.test_path != '':
            writer.close()

    # ------------------------------------------------------------------------
    # Saving the result for hyper parameter tuning 
    # ------------------------------------------------------------------------
    if args.mode == 'train_clf' and args.result_path:
        result = {
                'train_loss':      train_res['loss_lbl'],
                'train_acc':       train_res['acc'],
                'train_recall':    train_res['recall'],
                'train_precision': train_res['precision'],
                'train_f1':        train_res['f1'],
                'dev_loss':        dev_res['loss_lbl'], 
                'dev_acc':         dev_res['acc'],      
                'dev_recall':      dev_res['recall'],   
                'dev_precision':   dev_res['precision'],
                'dev_f1':          dev_res['f1'],       
                'test_loss':       test_res['loss_lbl'],
                'test_acc':        test_res['acc'],
                'test_recall':     test_res['recall'],
                'test_precision':  test_res['precision'],
                'test_f1':         test_res['f1'],
                'saved_path':      saved_path,
                }

        for attr, value in sorted(args.__dict__.items()):
            result[attr] = value

        with open(args.result_path, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
