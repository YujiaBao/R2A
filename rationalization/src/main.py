import argparse
import sys
from os.path import dirname, realpath
import data_utils as data_utils
import model_utils as model_utils
import train_utils as train_utils
import os
import torch
import datetime
import pickle

parser = argparse.ArgumentParser(description='Rationalizing Neural Predictions')
# learning
parser.add_argument('--dropout',    type=float, default=0.1, 
    help='Dropout zero probability [default: 0.1]')
parser.add_argument('--lr',         type=float, default=0.001, 
    help='initial learning rate [default: 0.001]')
parser.add_argument('--batch_size', type=int, default=50,
    help='batch size for training [default: 50]')
parser.add_argument('--patience',   type=int, default=5, 
    help='Divide the current learning rate by 10 when dev loss stop improving during the last ')

# model configuration
# classifier, use a CNN structure
parser.add_argument('--num_filters',  type=int, default=50, 
    help="Num of filters per filter size [default: 50]")
parser.add_argument('--filter_sizes', type=str, default="3,5,7", 
    help="Filter sizes [default: 3,4,5]")
parser.add_argument('--hidden_dim',   type=int, default=50, 
    help='Dim. of the hidden layer (between conv and output) [default: 100]')

# rationale generator 
parser.add_argument('--dependent',   action='store_true', default=False, help='dependent sampling')
parser.add_argument('--rnn_size',    type=int, default=200, 
    help='Dim. of the hidden state for the RNN [default: 200]')
parser.add_argument('--cell_type',   type=str, default="LSTM", 
    help="cell type: LSTM, GRU")

# regularization parameter
parser.add_argument('--temperature', type=float, default=1.0, 
    help='Temperature of the gumbel softmax. Anneal it after each epoch. [default: 10]')
parser.add_argument('--l_selection', type=float, default=0.05, 
    help='Penalize num. of words selected. [default: 1e-4]')
parser.add_argument('--l_variation', type=float, default=0.05,
    help='Encourage coherence of the selected rationale. [default: 2e-4]')
parser.add_argument('--l_selection_target', type=float, default=0.01,
    help='Desired probability of selection')


parser.add_argument('--mode',         type=str, default='train',
    help='Running mode. One of the following: train, test')
parser.add_argument('--test_path',    type=str, default='',
    help='Path to the testing data. Required for mode test.')

# data 
parser.add_argument('--num_classes', type=int, default=1, 
    help='Num of classe. If 1, then do regression. [default: 1]')

parser.add_argument('--dataset',     type=str, default='beer0', help='Choice of task')
parser.add_argument('--word_vector', type=str, default='fasttext.en.300d', 
    help='Name of pretrained word embeddings. Options: charngram.100d fasttext.en.300d ' \
    'fasttext.simple.300d glove.42B.300d glove.840B.300d glove.twitter.27B.25d ' \
    'glove.twitter.27B.50d glove.twitter.27B.100d glove.twitter.27B.200d glove.6B.50d ' \
    'glove.6B.100d glove.6B.200d glove.6B.300d [Default: fasttext.simple.300d]')
parser.add_argument('--fine_tune_wv',type=int, default=0,
    help='Set this to > 0 to fine tune word vectors.')

# option
parser.add_argument('--cuda',     action='store_true', default=False, help='run on gpu')
parser.add_argument('--filter',   action='store_true', default=False, help='filter discontinuities')

parser.add_argument('--save',     action='store_true', default=False, help='save model snapshot after training')
parser.add_argument('--dispatcher',action='store_true', default=False, help='Run from dispatcher or not')

parser.add_argument('--snapshot',    type=str, default=None, help='path for loading model snapshot [default: None]')
parser.add_argument('--result_path', type=str, default=None, 
                                     help='Path to store a pickle file of the resulting performance [default: None]')
parser.add_argument('--torch_seed', type=int, default=226,
    help='Path to store a pickle file of the resulting performance [default: 226 (bday)]')


args = parser.parse_args()
args.filter_sizes = [int(K) for K in args.filter_sizes.split(',')]
torch.manual_seed(args.torch_seed)

if __name__ == '__main__':
    # update args and print
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    if args.snapshot is not None:
        vocab = pickle.load(open(args.snapshot + '.vocab', 'rb'))
    else:
        vocab = None

    train_data, dev_data, test_data, vocab = data_utils.load_dataset(args)

    if args.mode == 'train':
        # Load model
        model = model_utils.get_model(vocab, args, args.snapshot)

        # train the network and early stop by dev loss
        dev_res = train_utils.train(train_data, dev_data, model, args)

        if args.save:
            with open(dev_res[-1]+'.vocab', 'wb') as f:
                pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

        prefix = 'results_rat/' + args.dataset + '/'
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        print("Evaluate on train set")
        writer = data_utils.generate_writer(
                prefix + 'sel_'+str(args.l_selection) + '_target_' + str(args.l_selection_target) +\
                '_var_' + str(args.l_variation) + '.train')
        train_res = train_utils.evaluate(train_data, model, args, writer)
        data_utils.close_writer(writer)

        print("Evaluate on dev set")
        writer = data_utils.generate_writer(
                prefix + 'sel_'+str(args.l_selection) + '_target_' + str(args.l_selection_target) +\
                '_var_' + str(args.l_variation) + '.dev')
        dev_res = train_utils.evaluate(dev_data, model, args, writer)
        data_utils.close_writer(writer)

        if args.result_path:
            result = {
                    'train_loss':      train_res[0],
                    'train_acc':       train_res[1],
                    'train_recall':    train_res[2],
                    'train_precision': train_res[3],
                    'train_f1':        train_res[4],
                    'dev_loss':        dev_res[0],
                    'dev_acc':         dev_res[1],
                    'dev_recall':      dev_res[2],
                    'dev_precision':   dev_res[3],
                    'dev_f1':          dev_res[4],
                    'saved_path':      dev_res[-1],
                    }

            for attr, value in sorted(args.__dict__.items()):
                result[attr] = value

            with open(args.result_path, 'wb') as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'test':
        # Load model
        model = model_utils.get_model(vocab, args, args.snapshot)

        print("Evaluate on test set")
        writer = data_utils.generate_writer(args.test_path + '.rationale')
        test_res = train_utils.evaluate(test_data, model, args, writer)
        data_utils.close_writer(writer)
