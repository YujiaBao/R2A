import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

def count_parameters(model_dict):
    '''
    Counting the total number of free (learnable) parameters of the model.
    
    Argument:
        model_dict: dict, a dictionary of models (see get_model for detailed description of each
        key-value pair)
    
    Return:
        total_sum: int, total number of parameters
    '''
    total_sum = 0
    for key in model_dict.keys():
        total_sum += sum(p.numel() for p in model_dict[key].parameters() if p.requires_grad)

    return total_sum

def save_model(model_dict, path):
    '''
    save model to a specified path
    
    Argument:
        model_dict: dict, a dictionary of models (see get_model for detailed description of each
        key-value pair)
        path: str, specify the location of the saved model.
    
    No return
    '''
    for key, model in model_dict.items():
        torch.save(model, path + '.' + key)
    print("Saved current best weights to {}\n".format(path))

def load_saved_model(path, args):
    '''
    load saved model from path
    
    Argument:
        path: str, specify the location of the saved model
        args: arguments (specify the settings)
    
    
    Return:
        model_dict: a dictionary of models (see get_model for detailed description of each
        key-value pair)
    '''
    try:
        model = {}
        model['critic']     = torch.load(path + '.critic')
        model['encoder']    = torch.load(path + '.encoder')
        model['r2a']        = torch.load(path + '.r2a')
        model['transform']  = torch.load(path + '.transform')

        # loading the task specific classifier (of the multitask learning module)
        for task in args.src_dataset:
            try:
                model[task] = torch.load(path + '.' + task)
            except Exception as e:
                # This happens at the step 3 of our pipeline, where we use the pre-trained encoder
                # to initialize the target classifier
                print(e)
                print('Randomly initialize head for task ' + task)
                model[task] = Classifier(args, task)

        if args.cuda:
            for key in model.keys():
                model[key].cuda()

        return model

    except Exception as e:
        print(e)
        raise ValueError("Failed to load the path.")

def get_model(vocab, args):
    '''
    Aim: initialize a model if no saved model snapshot is specified. Otherwise, load the snapshot.

    Argument:
        vocab: torchtext.vocab, vocabulary of the word embedding mapping
        args: arguments
    
    
    Return:
        model_dict: a dictionary of models (see get_model for detailed description of each
        key-value pair)
    '''

    if args.snapshot is None:
        # initialize a new model
        print("\nBuilding model...")
        model = {}
        model['critic']  = WassersteinD(args)   # Critic network for estimating the L_WD
        model['encoder'] = Encoder(vocab, args) # encoding the input into hidden representation
        model['transform'] = nn.Linear(args.rnn_size, args.rnn_size//2) # transformation layer
        model['r2a']     = R2A(args) # mapping rationale + hidden representation

        for task in args.src_dataset:
            model[task] = Classifier(args, task)

    else:
        # load saved model
        print('\nLoading model from %s' % args.snapshot)
        model = load_saved_model(args.snapshot, args)

    print("Total num. of parameters: %d" % count_parameters(model))

    if args.cuda:
        for key in model.keys():
            model[key].cuda()
        return model

    else:
        return model

class Embedding(nn.Module):
    def __init__(self, vocab, fine_tune_wv):
        '''
            This module aims to convert the token id into its corresponding embedding.
        '''
        super(Embedding, self).__init__()
        vocab_size, embedding_dim = vocab.vectors.size()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.weight.data = vocab.vectors
        self.embedding_layer.weight.requires_grad = False
        # do not finetune the embedding

        self.fine_tune_wv = fine_tune_wv
        if self.fine_tune_wv > 0:
            # if strictly positive, augment the original fixed embedding by a tunable embedding of
            # dimension fine_tune_wv
            self.tune_embedding_layer = nn.Embedding(vocab_size, self.fine_tune_wv)

        self.embedding_dim = embedding_dim + fine_tune_wv

    def forward(self, text):
        '''
            Argument:
                text:   batch_size * max_text_len
            Return:   
                output: batch_size * max_text_len * embedding_dim
        '''
        output = self.embedding_layer(text).float()

        if self.fine_tune_wv > 0:
            output = torch.cat([output, self.tune_embedding_layer(text).float()], dim=2)

        return output

class RNN(nn.Module):
    def __init__(self, cell_type, input_dim, hidden_dim, num_layers, bidirectional, dropout):
        '''
            This module is a wrapper of the RNN module
        '''
        super(RNN, self).__init__()

        if cell_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                    bidirectional=bidirectional, dropout=dropout)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                    bidirectional=bidirectional, dropout=dropout)
        else:
            raise ValueError('Only GRU and LSTM are supported')

    def _sort_tensor(self, input, lengths):
        ''' 
        pack_padded_sequence  requires the length of seq be in descending order to work.
        Returns the sorted tensor, the sorted seq length, and the indices for inverting the order.

        Input:
                input: batch_size, seq_len, *
                lengths: batch_size
        Output:
                sorted_tensor: batch_size-num_zero, seq_len, *
                sorted_len:    batch_size-num_zero
                sorted_order:  batch_size
                num_zero
        '''
        sorted_lengths, sorted_order = lengths.sort(0, descending=True)
        sorted_input = input[sorted_order]
        _, invert_order  = sorted_order.sort(0, descending=False)

        # Calculate the num. of sequences that have len 0
        nonzero_idx = sorted_lengths.nonzero()
        num_nonzero = nonzero_idx.size()[0]
        num_zero = sorted_lengths.size()[0] - num_nonzero

        # temporarily remove seq with len zero
        sorted_input = sorted_input[:num_nonzero]
        sorted_lengths = sorted_lengths[:num_nonzero]

        return sorted_input, sorted_lengths, invert_order, num_zero

    def _unsort_tensor(self, input, invert_order, num_zero):
        ''' 
        Recover the origin order

        Input:
                input:        batch_size-num_zero, seq_len, hidden_dim
                invert_order: batch_size
                num_zero  
        Output:
                out:   batch_size, seq_len, *
        '''
        if num_zero == 0:
            input = input.index_select(0, invert_order)

        else:
            dim0, dim1, dim2 = input.size()
            zero = torch.zeros(num_zero, dim1, dim2)
            if self.args.cuda:
                zero = zero.cuda()

            input = torch.cat((input, zero), dim=0)
            input = input[invert_order]

        return input

    def forward(self, text, text_len):
        '''
        Input: text, text_len
            text        batch_size * max_text_len * input_dim
            text_len    batch_size

        Output: text
            text        batch_size * max_text_len * output_dim
        '''
        # Go through the rnn
        # Sort the word tensor according to the sentence length, and pack them together
        sort_text, sort_len, invert_order, num_zero = self._sort_tensor(input=text, lengths=text_len)
        text = pack_padded_sequence(sort_text, lengths=sort_len.cpu().numpy(), batch_first=True)

        # Run through the word level RNN
        text, _ = self.rnn(text)         # batch_size, max_doc_len, args.word_hidden_size

        # Unpack the output, and invert the sorting
        text = pad_packed_sequence(text, batch_first=True)[0] # batch_size, max_doc_len, rnn_size
        text = self._unsort_tensor(text, invert_order, num_zero) # batch_size, max_doc_len, rnn_size

        return text

class Encoder(nn.Module):
    '''
    Shared encoder (consists of embedding, RNN, and LM)
    '''
    def __init__(self, vocab, args):
        super(Encoder, self).__init__()
        self.args = args

        # Initialize embedding for the words
        self.ebd = Embedding(vocab, args.fine_tune_wv)

        # Word level RNN
        self.rnn = RNN(args.cell_type, self.ebd.embedding_dim , args.rnn_size//2, args.num_layers,
                True, args.dropout)

        self.lm  = LM(args.rnn_size, args.lm_bins)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, text, text_len, lm):
        '''
        Input:
            text:     batch_size, max_text_len
            text_len: batch_size
        '''

        word = self.ebd(text)
        word = self.dropout(word)

        hidden = self.rnn(word, text_len)

        if lm:
            loss_lm = self.lm(hidden, text, self.args.l_lm)
        else:
            loss_lm = 0

        return hidden, loss_lm

class LM(nn.Module):
    '''
    language model
    '''
    def __init__(self, biRNN_size, out_dim):
        super(LM, self).__init__()
        self.in_dim = biRNN_size//2
        self.out_dim = out_dim

        self.fwd_fc = nn.Linear(self.in_dim, out_dim) # classifier based on forward information
        self.bkd_fc = nn.Linear(self.in_dim, out_dim) # classifier based on backward information

    def forward(self, hidden, text, l_lm):
        if l_lm != 0:
            if self.out_dim != 0:
                text = torch.remainder(text, self.out_dim)

            out_fwd = self.fwd_fc(hidden[:,:-1,:self.in_dim]).view(-1, self.out_dim)
            out_bkd = self.bkd_fc(hidden[:,1:,self.in_dim:]).view(-1, self.out_dim)

            gold_fwd = text[:,1:].contiguous().view(-1).detach()
            gold_bkd = text[:,:-1].contiguous().view(-1).detach()

            loss = F.cross_entropy(out_fwd, gold_fwd) + F.cross_entropy(out_bkd, gold_bkd)

            return loss

        else:
            return 0

class WassersteinD(nn.Module):
    '''
    Estimating the Wasserstein distance between two distributions
    '''
    def __init__(self, args):
        super(WassersteinD, self).__init__()

        self.args = args

        # parametrize the critic as a MLP
        self.seq = nn.Sequential(
            nn.Linear(args.rnn_size, args.rnn_size//2),
            nn.ReLU(),
            nn.Linear(args.rnn_size//2, 1)
        )

    def _aggregate_forward_backward(self, input, input_len):
        '''
        The input is a sequence of variable length. We convert it into a fixed-length representation
        by concatenating its first and last element.
        '''

        output = []
        for i, last in enumerate(input_len):
            output.append(torch.cat([input[i:i+1, last-1,:], input[i:i+1, 0,:]], dim=1))

        output = torch.cat(output, dim=0)

        return output

    def forward(self, src_x, src_len, tar_x, tar_len, wd_only):
        src = self._aggregate_forward_backward(src_x, src_len) # batch, args.rnn_size
        tar = self._aggregate_forward_backward(tar_x, tar_len) # batch, args.rnn_size

        wd_loss = self.seq(src).mean() - self.seq(tar).mean()

        if wd_only: # just return the L_WD (used for inference of the critic)
            return wd_loss, 0

        # Computing the gradient penalty (used for the training of the critic)

        # Sample between the source and the target. For gradient penalty
        eps = torch.FloatTensor(src.shape[0], 1).uniform_(0., 1.)
        if src.is_cuda:
            eps = eps.cuda()

        x_hat = tar + eps * (src - tar)
        x_hat.requires_grad = True

        disc_x_hat = self.seq(x_hat)

        if src.is_cuda:
            gradients = autograd.grad(outputs=disc_x_hat, inputs=x_hat,
                                      grad_outputs=torch.ones(disc_x_hat.size()).cuda(),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
        else:
            gradients = autograd.grad(outputs=disc_x_hat, inputs=x_hat,
                                      grad_outputs=torch.ones(disc_x_hat.size()),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return wd_loss, gradient_penalty

class Attention(nn.Module):
    '''
    Computing the attention over the words
    '''
    def __init__(self, input_dim, proj_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.proj_dim  = proj_dim

        self.head = nn.Parameter(torch.Tensor(proj_dim, 1).uniform_(-0.1, 0.1))
        self.proj = nn.Linear(input_dim, proj_dim)

    def forward(self, input, input_mask):
        '''
        input: batch, max_text_len, input_dim
        input_mask: batch, max_text_len
        '''
        batch, max_input_len, input_dim = input.size()

        proj_input = torch.tanh(self.proj(input.view(batch*max_input_len, -1)))
        att = torch.mm(proj_input, self.head)
        att = att.view(batch, max_input_len, 1)
        log_att = F.log_softmax(att, dim=1)
        att = F.softmax(att, dim=1)

        output = input * att * input_mask.unsqueeze(-1).detach()
        output = output.sum(dim=1)

        return output, att.squeeze(2), log_att.squeeze(2)

class Classifier(nn.Module):
    '''
    Task specific classifier (consisting of attention and MLP
    '''
    def __init__(self, args, task):
        super(Classifier, self).__init__()
        self.args = args

        # attention
        self.attention = Attention(args.rnn_size, args.proj_dim)

        # Fully connected hidden layers
        if args.hidden_dim == 0:
            self.seq = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.rnn_size, args.num_classes[task])
            )
        else:
            self.seq = nn.Sequential(
                nn.Linear(args.rnn_size, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, args.num_classes[task])
            )

    def forward(self, hidden, text_mask):
        """
        Input:
            hidden:     batch_size, max_text_len, rnn_size
            text_mask:  batch_size, max_text_len
        """
        out, att, log_att = self.attention(hidden, text_mask)

        out = self.seq(out)

        return out, att, log_att

class R2A(nn.Module):
    '''
    This is the attention generation module.
    It combines task-specific rationale information with the input information to generate the
    attention.
    '''
    def __init__(self, args):
        super(R2A, self).__init__()
        self.args = args
        self.rnn = RNN(args.cell_type, args.rnn_size//2+2, args.proj_dim//2, 1, True, args.dropout)
        self.att = Attention(args.proj_dim, args.proj_dim)

    def forward(self, text, rationale, rat_freq, text_len, text_mask):
        input = torch.cat([text, rationale.unsqueeze(-1), rat_freq.unsqueeze(-1)], dim=2)
        hidden = self.rnn(input, text_len)
        out, att, log_att = self.att(hidden, text_mask)

        return att, log_att
