import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
# from gumbel import Gumbel

torch.manual_seed(226)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model(vocab, args, snapshot):
    if snapshot is None:
        # initialize a new model
        print("\nBuilding model...")
        model = TAO(vocab, args)

    else:
        # load saved model
        print('\nLoading model from %s' % snapshot)
        try:
            model = torch.load(args.snapshot)

        except :
            raise ValueError("Snapshot doesn't exist.")

    print("Total num. of parameters: %d" % count_parameters(model))

    if args.cuda:
        return model.cuda()
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

class TAO(nn.Module):

    def __init__(self, vocab, args):
        super(TAO, self).__init__()
        self.args = args

        # Word embedding
        self.ebd = Embedding(vocab, args.fine_tune_wv)

        # Generator RNN
        self.rnn = RNN(args.cell_type, self.ebd.embedding_dim , args.rnn_size//2, 1, True,
                args.dropout)

        if args.dependent:
            if args.cell_type == 'GRU':
                self.rnn_sample = nn.GRU(args.rnn_size+1, args.rnn_size//2, bidirectional=False)
            elif args.cell_type == 'LSTM':
                self.rnn_sample = nn.LSTM(args.rnn_size+1, args.rnn_size//2, bidirectional=False)

            self.gen_fc = nn.Linear(args.rnn_size//2, 2)

        else:
            self.gen_fc = nn.Linear(args.rnn_size, 2)

        # Classifier
        # CNN
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.ebd.embedding_dim, 
            out_channels=args.num_filters, kernel_size=K) for K in args.filter_sizes])
        self.num_filters_total = args.num_filters * len(args.filter_sizes)

        # Fully connected
        if args.hidden_dim != 0:
            self.seq = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.num_filters_total, args.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(args.hidden_dim, args.num_classes)
                    )

        else:
            self.seq = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(args.num_filters_total, args.num_classes)
                    )

        self.dropout = nn.Dropout(args.dropout)

    def _gumbel_softmax(self, logit, temperature, cuda):
        '''
        generate a gumbel softmax based on the logit
        noise is a sample from gumbel(0,1)
        '''
        eps = 1e-20
        noise = torch.rand(logit.size()).cuda()
        noise = - torch.log(-torch.log(noise+eps)+eps)
        x = (logit + noise.detach()) / temperature
        return F.softmax(x, dim=-1)

    def _conv_max_pool(self, x, conv_filter):
        '''
        Compute sentence level convolution
        Input:
            x:      batch_size, max_doc_len, embedding_dim
        Output:     batch_size, num_filters_total
        '''
        assert(len(x.size()) == 3)

        x = x.permute(0,2,1)                     # batch_size, embedding_dim, doc_len
        x = x.contiguous()
        x = [conv(x) for conv in conv_filter]     # [batch_size, num_filters, doc_len-filter_size+1] * len(filter_size)
        x = [F.max_pool1d(sub_x, sub_x.size(2)).squeeze(2) for sub_x in x] # [batch_size, num_filters] * len(filter_size)
        x = torch.cat(x, 1)                      # batch_size, num_filters_total

        return x

    def _independent_sampling(self, x, temperature, hard):
        '''
        Use the hidden state at all time to sample whether each word is a rationale or not.
        No dependency between actions.
        Return the sampled soft rationale mask
        '''
        rationale_logit = F.log_softmax(self.gen_fc(x), dim=2)
        rationale_mask  = self._gumbel_softmax(rationale_logit, temperature, self.args.cuda)

        # extract the probability of being a rationale from the two dimensional gumbel softmax
        rationale_mask  = rationale_mask[:,:,1]

        if hard: # replace soft mask by hard mask, no longer differentiable, only used during testing
            rationale_mask = torch.round(rationale_mask).float()

        return rationale_mask

    def _dependent_sampling(self, x, temperature, hard):
        '''
            Use the hidden state at all time to sample one by one using another RNN.
            Return the sampled soft rationale mask
            x: batch, text_len, hidden_size
        '''
        batch_size, max_text_len, _ = x.size()

        # Sample the rationale label for the first token
        xn = torch.cat([x[:,0,:], torch.zeros(batch_size,1).cuda()], dim=1).unsqueeze(0) # 1, batch * (rnn_size + 1)
        out, hn = self.rnn_sample(xn)

        logit  = F.log_softmax(self.gen_fc(out[0]), dim=-1) # batch, 2
        mask   = self._gumbel_softmax(logit, temperature, self.args.cuda) # batch, 2
        mask   = torch.round(mask[:,1]).float() if hard else mask[:,1]  # batch
        mask   = mask.unsqueeze(1)  # batch, 1

        rationale_mask = [mask]

        # go through the sequence one by one
        for i in range(1, max_text_len):
            xn = torch.cat([x[:,i,:], rationale_mask[-1].detach()], dim=1).unsqueeze(0) # 1, batch * (rnn_size+1) 
            out, hn = self.rnn_sample(xn, hn)

            logit  = F.log_softmax(self.gen_fc(out[0]), dim=-1) # batch, 2
            mask   = self._gumbel_softmax(logit, temperature, self.args.cuda) # batch, 2
            mask   = torch.round(mask[:,1]).float() if hard else mask[:,1]  # batch
            mask   = mask.unsqueeze(1)  # batch, 1
            rationale_mask.append(mask)

        return torch.cat(rationale_mask, dim=1)

    def forward(self, text, text_len, temperature, hard=False):
        word = self.ebd(text)
        word = self.dropout(word)

        # Generator
        hidden = self.rnn(word, text_len)

        # Sample rationale indicator
        if self.args.dependent:
            rationale = self._dependent_sampling(hidden, temperature, hard)
        else:
            rationale = self._independent_sampling(hidden, temperature, hard)

        # mask out non-rationale words
        rat_word = word * rationale.unsqueeze(2)  # batch, len, embedding_dim

        # apply conv filters
        hidden = self._conv_max_pool(rat_word, self.convs)  # batch, num_filters_total

        # run the MLP
        out = self.seq(hidden)

        return out, rationale
