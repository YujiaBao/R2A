import os
import sys
import torch
import torch.autograd as autograd
from itertools import chain
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import data_utils as data_utils
import model_utils as model_utils
import datetime
import sklearn.metrics as metrics
import time
import numpy as np
from termcolor import colored
import math
from tqdm import tqdm

def _to_tensor(x_list, cuda=True):
    '''
    Convert a list of numpy arrays into a list of pytorch tensors
    '''
    if type(x_list) is not list:
        x_list = [x_list]

    res_list = []
    for x in x_list:
        x = torch.from_numpy(x)
        if cuda:
            x = x.cuda()
        res_list.append(x)

    if len(res_list) == 1:
        return res_list[0]
    else:
        return tuple(res_list)

def _to_numpy(x_list):
    '''
    Convert a list of tensor into a list of numpy arrays
    '''
    if type(x_list) is not list:
        x_list = [x_list]

    res_list = []
    for x in x_list:
        res_list.append(x.data.cpu().numpy())

    if len(res_list) == 1:
        return res_list[0]
    else:
        return tuple(res_list)

def _to_number(x):
    '''
    Convert a scalar tensor into a python number
    '''
    if isinstance(x, torch.Tensor):
        return x.item()
    else:
        return x

def _compute_score(y_pred, y_true, num_classes=2):
    '''
    Compute the accuracy, f1, recall and precision
    '''
    if num_classes == 2:
        average = "binary"
    else:
        average = "macro"

    acc       = metrics.accuracy_score( y_pred=y_pred, y_true=y_true)
    f1        = metrics.f1_score(       y_pred=y_pred, y_true=y_true, average=average)
    recall    = metrics.recall_score(   y_pred=y_pred, y_true=y_true, average=average)
    precision = metrics.precision_score(y_pred=y_pred, y_true=y_true, average=average)

    return acc, f1, recall, precision

def _init_optimizer(model, args):
    '''
    initialize the optimizer and the learning rate scheduler of the model.
    '''
    optimizer = {}
    optimizer['critic'] = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model['critic'].parameters()) , lr=args.lr)
    optimizer['encoder'] = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model['encoder'].parameters()) , lr=args.lr)
    optimizer['r2a'] = torch.optim.Adam(
        filter(lambda p: p.requires_grad,
            chain(model['r2a'].parameters(), model['transform'].parameters())), lr=args.lr)

    for task in args.src_dataset:
        if task != '':
            optimizer[task] = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model[task].parameters()) , lr=args.lr)

    scheduler = {}
    # Initialize the learning rate scheduler for the encoder (that is shared by all other modules)
    scheduler['encoder'] = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer['encoder'], 'min', patience=args.patience, factor=0.1, verbose=True)

    # Initialize the learning rate scheduler for The R2A (Attention generation module)
    scheduler['r2a'] = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer['r2a'], 'min', patience=args.patience, factor=0.1, verbose=True)

    # Initialize the learning rate scheduler for the task specific classifier
    for task in args.src_dataset:
        if task != '':
            scheduler[task] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer[task], 'min', patience=args.patience, factor=0.1, verbose=True)

    return optimizer, scheduler

def _print_train_res(train_res, args):
    print('=== TRAIN ===')
    for task in args.src_dataset:
        print("{:15s} {:s} {:.2f} {:s} {:.4f}, {:s} {:.4f} * {:.1e} {:s} {:.4f} * {:.1e}, {:s} {:.4f} * {:.1e},"\
                " {:s} {:.4f} * {:.1e}, {:s} {:.4f} * {:.1e}".format(
                task,
                colored("ep:", "cyan"),
                train_res[-1][task]['epoch'],
                colored("l_lbl", "red"),
                sum([res[task]['loss_lbl'] for res in train_res])/len(train_res),
                colored("l_wd ", "red"),
                sum([res[task]['loss_wd'] for res in train_res])/len(train_res),
                args.l_wd,
                colored("l_src_lm ", "red"),
                sum([res[task]['loss_src_lm'] for res in train_res])/len(train_res),
                args.l_lm,
                colored("l_tar_lm ", "red"),
                sum([res[task]['loss_tar_lm'] for res in train_res])/len(train_res),
                args.l_lm,
                colored("l_r2a", "red"),
                sum([res[task]['loss_r2a'] for res in train_res])/len(train_res),
                args.l_r2a,
                colored("l_a2r", "red"),
                sum([res[task]['loss_a2r'] for res in train_res])/len(train_res),
                args.l_a2r,
                ))

def print_dev_res(dev_res, args):
    loss_tot     = sum([res['loss_total'] for res in dev_res])/len(dev_res)
    loss_lbl     = sum([res['loss_lbl'] for res in dev_res])/len(dev_res)
    loss_wd      = sum([res['loss_wd'] for res in dev_res])/len(dev_res)
    loss_r2a     = sum([res['loss_r2a'] for res in dev_res])/len(dev_res)
    loss_lbl_r2a = sum([res['loss_lbl_r2a'] for res in dev_res])/len(dev_res)
    loss_a2r     = sum([res['loss_a2r'] for res in dev_res])/len(dev_res)
    loss_encoder = sum([res['loss_encoder'] for res in dev_res])/len(dev_res)
    loss_src_lm  = sum([res['loss_src_lm'] for res in dev_res])/len(dev_res)
    loss_tar_lm  = sum([res['loss_tar_lm'] for res in dev_res])/len(dev_res)
    acc          = sum([res['acc'] for res in dev_res])/len(dev_res)
    recall       = sum([res['recall'] for res in dev_res])/len(dev_res)
    f1           = sum([res['f1'] for res in dev_res])/len(dev_res)
    precision    = sum([res['precision'] for res in dev_res])/len(dev_res)

    print("{:15s} {:s} {:.4f}, {:s} {:.4f} * {:.1e}, {:s} {:.4f} * {:.1e}, {:s} {:.4f} * {:.1e}, {:s} {:.4f} * {:.1e}, "\
            "{:s} {:.4f} * {:.1e}".format(
            'overall',
            colored("l_lbl", "red"),
            loss_lbl,
            colored("l_wd ", "red"),
            loss_wd,
            args.l_wd,
            colored("l_src_lm ", "red"),
            loss_src_lm,
            args.l_lm,
            colored("l_tar_lm ", "red"),
            loss_tar_lm,
            args.l_lm,
            colored("l_r2a", "red"),
            loss_r2a,
            args.l_r2a,
            colored("l_a2r", "red"),
            loss_a2r,
            args.l_a2r,
            ))

    return {
            'loss_total': loss_tot,
            'loss_lbl':   loss_lbl,
            'loss_lbl_r2a':   loss_lbl_r2a,
            'loss_wd':    loss_wd,
            'loss_r2a':   loss_r2a,
            'loss_a2r':   loss_a2r,
            'loss_src_lm':   loss_src_lm,
            'loss_tar_lm':   loss_tar_lm,
            'loss_encoder':loss_encoder,
            'acc':        acc,
            'recall':     recall,
            'f1':         f1,
            'precision':  precision,
            }

def train(train_data, dev_data, model, args):
    timestamp = str(int(time.time() * 1e7))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "tmp-runs", timestamp))
    print("Saving the model to {}\n".format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best = 100
    best_path = ""
    sub_cycle = 0

    optimizer, scheduler = _init_optimizer(model, args)

    tar_train_batches = None if (args.mode == 'train_clf' or args.mode == 'test_clf') else \
            data_utils.data_loader(train_data[args.tar_dataset], args.batch_size, oneEpoch=False)
    src_unlbl_train_batches = None if (args.mode == 'train_clf' or args.mode == 'test_clf') else \
            data_utils.data_loader(train_data[args.src_dataset[0]], args.batch_size, oneEpoch=False)
    src_train_batches = data_utils.data_dict_loader(train_data, args.src_dataset, args.batch_size)

    tar_dev_data = None if args.tar_dataset == '' else dev_data[args.tar_dataset]

    ep = 1
    while True:
        start = time.time()

        train_res = []

        if args.dispatcher:
            for i in range(args.epoch_size):
                cur_res = train_batch(
                        model, next(src_train_batches), src_unlbl_train_batches, tar_train_batches, optimizer, args)
                train_res.append(cur_res)

        else:
            for batch in tqdm(range(args.epoch_size), dynamic_ncols=True):
                cur_res = train_batch(
                        model, next(src_train_batches), src_unlbl_train_batches, tar_train_batches, optimizer, args)
                train_res.append(cur_res)

        end = time.time()
        print("\n{}, Updates {:5d}, Time Cost: {} seconds".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            ep*args.epoch_size, end-start))

        _print_train_res(train_res, args)

        # evaluate on dev set
        print('===  DEV  ===')
        dev_res = []
        for task in args.src_dataset:
            writer = open(os.path.join(out_dir, str(ep)) + '.' + task + '.out', 'w')
            cur_res = evaluate_task(
                    dev_data[task], task, tar_dev_data, model, optimizer, args, writer=writer)
            writer.close()
            dev_res.append(cur_res)

            scheduler[task].step(cur_res['loss_lbl'])

        dev_res = print_dev_res(dev_res, args)

        # adjust the encoder loss based on avg. loss lbl plus avg. loss wd
        scheduler['encoder'].step(dev_res['loss_encoder'])

        # adjust the encoder loss based on avg. r2a loss
        scheduler['r2a'].step(dev_res['loss_r2a'])

        if (args.mode != 'train_clf' and dev_res['loss_lbl_r2a'] < best) or\
                (args.mode == 'train_clf' and dev_res['loss_lbl'] < best):

            best = dev_res['loss_lbl_r2a'] if args.mode != 'train_clf' else dev_res['loss_lbl']

            best_path = os.path.join(out_dir, str(ep))
            model_utils.save_model(model, best_path)
            sub_cycle = 0
        else:
            sub_cycle += 1

        if sub_cycle == args.patience*2:
            break

        ep += 1

    print("End of training. Restore the best weights")
    model = model_utils.load_saved_model(best_path, args)

    print('===  BEST DEV  ===')
    dev_res = []
    for task in args.src_dataset:
        cur_res = evaluate_task(
                dev_data[task], task, tar_dev_data, model, None, args)
        dev_res.append(cur_res)

    dev_res = print_dev_res(dev_res, args)

    print("Deleting model snapshot")
    os.system("rm -rf {}/*".format(out_dir)) # delete model snapshot for space

    if args.save:
        print("Save the best model to director saved-runs")
        best_dir = os.path.abspath(os.path.join(os.path.curdir, "saved-runs", args.mode, \
            "-".join(args.src_dataset) + '_' + args.tar_dataset + '_' + timestamp))

        if not os.path.exists(best_dir):
            os.makedirs(best_dir)

        best_dir = os.path.join(best_dir, 'best')
        model_utils.save_model(model, best_dir)

        with open(best_dir+'_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr.upper(), value))

        return dev_res, best_dir, model

    return dev_res, out_dir, model

def _get_mask(text_len, cuda):
    idxes = torch.arange(0, int(torch.max(text_len)), 
                         out=torch.LongTensor(torch.max(text_len).item())).unsqueeze(0)

    if cuda:
        idxes = idxes.cuda()

    text_mask = (idxes < text_len.unsqueeze(1)).float().detach() # batch, text_len

    return text_mask

def train_batch(model, src_batch, src_unlbl_batches, tar_batches, optimizer, args):
    '''
        Train the network on a batch of examples

        model: a dictionary of networks
        src_batch: a dictionary of data for each task (each value is a batch of examples)
        src_unlbl_batches: an iterator that generates a batch of source examples (used for training
            the domain-invariant encoder)
        tar_unlbl_batches: an iterator that generates a batch of target examples (used for training
            the domain-invariant encoder)
        optimizer: the optimizer that updates the network weights
        args: the overall argument
    '''
    # ------------------------------------------------------------------------
    # Step 1:  Training the critic network
    # ------------------------------------------------------------------------
    # set all network to eval mode except the critic.
    for key in model.keys():
        model[key].eval()
        if key in optimizer:
            optimizer[key].zero_grad()

    # train critic network for critic_steps
    if args.l_wd != 0 and args.mode == 'train_r2a':
        model['critic'].train()
        i = 0
        while True:
            # get target and source input, text only, no labels
            tar_text, _,  _, _, _, tar_text_len, _, _, _ = next(tar_batches)
            tar_text, tar_text_len = _to_tensor([tar_text, tar_text_len], args.cuda)
            src_text, _,  _, _, _, src_text_len, _, _, _ = next(src_unlbl_batches)
            src_text, src_text_len = _to_tensor([src_text, src_text_len], args.cuda)

            # run the encoder
            tar_hidden, _ = model['encoder'](tar_text, tar_text_len, False)
            src_hidden, _ = model['encoder'](src_text, src_text_len, False)

            # apply the transformation layer
            invar_tar_hidden = model['transform'](tar_hidden)
            invar_src_hidden = model['transform'](src_hidden)

            # run the critic network
            optimizer['critic'].zero_grad()
            loss_wd, grad_penalty = model['critic'](
                    invar_src_hidden.detach(), src_text_len,
                    invar_tar_hidden.detach(), tar_text_len, False)
            loss = -loss_wd + args.l_grad_penalty * grad_penalty

            # backprop
            loss.backward()
            optimizer['critic'].step()

            # by definition, loss_wd should be non-negative. If it is negative, it means the critic
            # network is not good enough. Thus, we need to train it more.
            i += 1
            if i >= args.critic_steps and _to_number(loss_wd) > 0:
                break

    # ------------------------------------------------------------------------
    # Step 2:  Training all the other networks
    # ------------------------------------------------------------------------
    # set all network to train mode except the critic
    for key in model.keys():
        model[key].train()
        if key in optimizer:
            optimizer[key].zero_grad()
    model['critic'].eval()

    # go though all source tasks
    result = {}
    for task, batch in src_batch.items():
        # get batch for task 
        text, rat_freq, rationale, gold_att, pred_att, text_len, label, _, epoch = batch

        # converting numpy arrays to pytorch tensor
        text, text_len, rat_freq, rationale, gold_att, pred_att, label = _to_tensor(
                [text, text_len, rat_freq, rationale, gold_att, pred_att, label], args.cuda)
        text_mask = _get_mask(text_len, args.cuda)

        # Run the encoder on source
        hidden, loss_src_lm = model['encoder'](text, text_len, True)
        invar_hidden = model['transform'](hidden)

        # Estimate l_wd
        loss_tar_lm = 0
        if args.l_wd != 0 and args.mode == 'train_r2a':
            # Run the encoder on target
            tar_text, _,  _, _, _, tar_text_len, _, _, _ = next(tar_batches)
            tar_text, tar_text_len = _to_tensor([tar_text, tar_text_len], args.cuda)

            tar_hidden, loss_tar_lm = model['encoder'](tar_text, tar_text_len, True)
            invar_tar_hidden = model['transform'](tar_hidden)

            loss_wd, _ = model['critic'](invar_hidden, text_len, invar_tar_hidden, tar_text_len, True)

        else:
            loss_wd = 0


        # Run the task-specific classifier
        # ------------------------------------------------------------------------
        if args.mode == 'train_clf' and not args.fine_tune_encoder:
            hidden = hidden.detach()

        out, att, log_att = model[task](hidden, text_mask)

        if args.num_classes[task] == 1:
            loss_lbl = F.mse_loss(torch.sigmoid(out.squeeze(1)), label)
        else:
            loss_lbl = F.cross_entropy(out, label)

        if  _to_number(torch.min(torch.sum(rationale, dim=1))) < 0.5: 
            # no words are annotated as rationale, add a small eps to avoid numerical error
            rationale = rationale + 1e-6

        # normalize the rationale score by the number of tokens in the document
        normalized_rationale = rationale * text_mask
        normalized_rationale = normalized_rationale / torch.sum(normalized_rationale, dim=1,
                keepdim=True)

        if args.mode == 'train_clf':
            # in this case, pred_att is loaded from the generated file and it provide supervision 
            # for the attention of the classifier
            if args.att_target == 'gold_att':
                pred_att = gold_att
            elif args.att_target == 'rationale':
                pred_att = normalized_rationale
            elif args.att_target == 'pred_att':
                pred_att = pred_att
            else:
                raise ValueError('Invalid supervision type.')

            log_pred_att = torch.log(pred_att)

        elif args.mode == 'train_r2a':
            # in this case, att (which is derived from the source multitask learning module)
            # is the supervision target for pred_att
            pred_att, log_pred_att = model['r2a'](
                    invar_hidden, rationale, rat_freq, text_len, text_mask)

        else:
            raise ValueError('Invalid mode')

        loss_a2r = 0
        loss_r2a = 1 - torch.mean(F.cosine_similarity(att, pred_att))

        if args.mode == 'train_r2a': 
            # only apply consistency regularization during r2a training
            loss_a2r = 1 - torch.mean(F.cosine_similarity(pred_att, normalized_rationale))

            # only apply hinge loss during r2a training
            if _to_number(loss_r2a) < 0.1:
                loss_r2a = 0
            else:
                loss_r2a = loss_r2a - 0.1

            if _to_number(loss_a2r) < 0.1:
                loss_a2r = 0
            else:
                loss_a2r = loss_a2r - 0.1


        loss = loss_lbl + args.l_r2a * loss_r2a + args.l_wd * loss_wd + \
                args.l_lm * (loss_src_lm + loss_tar_lm) + args.l_a2r * loss_a2r

        loss.backward()

        # update task specific parameters
        optimizer[task].step()

        # saved the performance on this batch
        result[task] = {
                'loss_lbl':   _to_number(loss_lbl),
                'loss_r2a':   _to_number(loss_r2a),
                'loss_wd':    _to_number(loss_wd),
                'loss_src_lm': _to_number(loss_src_lm),
                'loss_tar_lm': _to_number(loss_tar_lm),
                'loss_a2r'   : _to_number(loss_a2r),
                'epoch':      epoch,
                }

    # update shared parameters
    optimizer['encoder'].step()
    optimizer['r2a'].step()

    return result

def evaluate_batch(model, optimizer, task, batch, src_batches, tar_batches, args, writer=None):
    '''
        Evaluate the network on a batch of examples

        model: a dictionary of networks
        optimizer: the optimizer that updates the network weights
        task: the name of the task
        batch: a batch of examples for the specified task
        src_batches: an iterator that generates a batch of source examples (used for estimating the
            wasserstein distance)
        tar_batches: an iterator that generates a batch of source examples (used for estimating the
            wasserstein distance)
        args: the overall argument
        writer: a file object. If not none, will write the prediction result and the generated
            attention to the file
    '''
    # ------------------------------------------------------------------------
    # Step 1:  Training the critic network
    # ------------------------------------------------------------------------
    # set all network to eval mode except the critic.
    for key in model.keys():
        model[key].eval()
        if key in optimizer:
            optimizer[key].zero_grad()

    # train critic network for critic_steps
    if args.l_wd != 0 and (args.mode == 'train_r2a' or args.mode == 'test_r2a'):
        model['critic'].train()
        i = 0
        while True:
            # get target and source input, text only, no labels
            tar_text, _,  _, _, _, tar_text_len, _, _, _ = next(tar_batches)
            tar_text, tar_text_len     = _to_tensor([tar_text, tar_text_len], args.cuda)
            src_text, _,  _, _, _, src_text_len, _, _, _ = next(src_batches)
            src_text, src_text_len     = _to_tensor([src_text, src_text_len], args.cuda)

            # run the encoder
            tar_hidden, _ = model['encoder'](tar_text, tar_text_len, False)
            src_hidden, _ = model['encoder'](src_text, src_text_len, False)

            # apply the transformation layer
            invar_tar_hidden = model['transform'](tar_hidden)
            invar_src_hidden = model['transform'](src_hidden)

            # run the critic network
            optimizer['critic'].zero_grad()
            loss_wd, grad_penalty = model['critic'](
                    invar_src_hidden.detach(), src_text_len,
                    invar_tar_hidden.detach(), tar_text_len, False)
            loss = -loss_wd + args.l_grad_penalty * grad_penalty

            # backprop
            loss.backward()
            optimizer['critic'].step()

            # by definition, loss_wd should be non-negative. If it is negative, it means the critic
            # network is not good enough. Thus, we need to train it more.
            i += 1
            if i >= args.critic_steps and _to_number(loss_wd) > 0:
                break

    model['critic'].eval()

    # ------------------------------------------------------------------------
    # Step 2: Run all other networks
    # ------------------------------------------------------------------------

    # get the current batch
    text, rat_freq, rationale, gold_att, pred_att, text_len, label, raw, _ = batch

    # convert to variable and tensor
    text, text_len, rat_freq, rationale, gold_att, pred_att, label = _to_tensor(
            [text, text_len, rat_freq, rationale, gold_att, pred_att, label], args.cuda)
    text_mask = _get_mask(text_len, args.cuda)

    # Run the encoder on source
    hidden, loss_src_lm = model['encoder'](text, text_len, True)
    invar_hidden = model['transform'](hidden)
    loss_src_lm = np.ones(len(raw)) * _to_number(loss_src_lm)

    # Estimating l_wd
    loss_tar_lm = 0
    if args.l_wd != 0 and (args.mode == 'test_r2a' or args.mode == 'train_r2a'):
        # Run the encoder on target
        tar_text, _,  _, _, _, tar_text_len, _, _, _ = next(tar_batches)

        # truncate to match the src batch size
        tar_text_len = tar_text_len[:len(raw)]
        tar_text     = tar_text[:len(raw),:max(tar_text_len)]

        # convert to tensor and variable
        tar_text, tar_text_len     = _to_tensor([tar_text, tar_text_len], args.cuda)

        # run the encoder on target
        tar_hidden, loss_tar_lm = model['encoder'](tar_text, tar_text_len, True)
        invar_tar_hidden = model['transform'](tar_hidden)

        loss_wd, _ = model['critic'](invar_hidden, text_len, invar_tar_hidden, tar_text_len, True)
        loss_wd = np.ones(len(raw)) * _to_number(loss_wd)

    else:
        loss_wd = np.zeros(len(raw))

    loss_tar_lm = np.ones(len(raw)) * _to_number(loss_tar_lm)

    # Classifier
    out, att, log_att = model[task](hidden, text_mask)

    if args.num_classes[task] == 1:
        loss_lbl = _to_numpy(F.mse_loss(torch.sigmoid(out.squeeze(1)), label, reduce=False))
        pred_lbl = _to_numpy(torch.sigmoid(out.squeeze(1)))
    else:
        loss_lbl = _to_numpy(F.cross_entropy(out, label, reduce=False))
        pred_lbl = np.argmax(_to_numpy(out), axis=1)

    true_lbl = _to_numpy(label)

    if _to_number(torch.min(torch.sum(rationale, dim=1))) < 0.5:
        # no words are annotated as rationale, add a small eps to avoid numerical error
        rationale = rationale + 1e-6

    # normalize the rationale score by the number of tokens in the document
    normalized_rationale = rationale * text_mask
    normalized_rationale = normalized_rationale / torch.sum(normalized_rationale, dim=1,
            keepdim=True)

    if args.mode == 'train_clf' or args.mode == 'test_clf':
        # in this case, pred_att is loaded from the generated file and it provide supervision 
        # for the attention of the classifier
        if args.att_target == 'gold_att':
            target = gold_att
        elif args.att_target == 'rationale':
            target = normalized_rationale
        elif args.att_target == 'pred_att':
            target = pred_att
        else:
            raise ValueError('Invalid supervision type.')

        log_pred_att = torch.log(pred_att)

    elif args.mode == 'train_r2a':
        # in this case, att (which is derived from the source multitask learning module)
        # is the supervision target for pred_att
        pred_att, log_pred_att = model['r2a'](
                invar_hidden, rationale, rat_freq, text_len, text_mask)

    else:
        raise ValueError('Invalid mode')

    loss_a2r = 1 - F.cosine_similarity(att, normalized_rationale)
    loss_a2r = _to_numpy(loss_a2r)

    loss_r2a = 1 - F.cosine_similarity(att, pred_att)
    loss_r2a = _to_numpy(loss_r2a)


    # Write attention to a tsv file
    if writer:
        gold_att, rationale, pred_att, rat_freq = _to_numpy([att, rationale, pred_att, rat_freq])
        data_utils.write(writer, task, raw, true_lbl, gold_att, rationale, pred_att, rat_freq)

    return {
            'true_lbl': true_lbl,
            'pred_lbl': pred_lbl,
            'loss_r2a': loss_r2a,
            'loss_lbl': loss_lbl,
            'loss_wd' : loss_wd,
            'loss_a2r': loss_a2r,
            'loss_src_lm': loss_src_lm,
            'loss_tar_lm': loss_tar_lm,
            }

def evaluate_task(data, task, tar_data, model, optimizer, args, writer=None):
    '''
        For mode = train_r2a and test_r2a,
            evaluate the network on a test data for a source task.
        For mode = train_clf,
            evaluate the network on a test data for the current task.

        data: the data to be tested on
        task: the task of the data
        tar_data: target data, used for evaluating l_wd when mode=train_r2a or test_r2a
        model: a dictionary of networks
        optimizer: the optimizer that updates the network weights, it can be none. Used for
            estimating l_wd
        args: the overall argument
        writer: a file object. If not none, will write the prediction result and the generated
            attention to the file
    '''

    # write the header of the output file
    if writer:
        writer.write('task\tlabel\traw\trationale\tpred_att\tgold_att\trat_freq\n')

    # initialize the optimizer
    if optimizer is None:
        optimizer = {}
        optimizer['critic'] = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model['critic'].parameters()) , lr=args.lr)

    for key in model.keys():
        model[key].eval()

    # if training or testing r2a, use target data to evaluate l_wd
    tar_batches = None if args.tar_dataset == '' else\
            data_utils.data_loader(tar_data, args.batch_size, oneEpoch=False)
    src_batches = None if args.tar_dataset == '' else\
            data_utils.data_loader(data, args.batch_size, oneEpoch=False)

    # obtain an iterator to go through the test data
    batches = data_utils.data_loader(data, args.batch_size, shuffle=False)

    total = {}

    # Iterate over the test data. Concatenate all the results.
    for batch in batches:
        cur_res = evaluate_batch(model, optimizer, task, batch, src_batches, tar_batches, args, writer)

        # store results of current batch
        for key, value in cur_res.items():
            if key not in total:
                total[key] = value
            else:
                total[key] = np.concatenate((total[key], value))

    # average loss across all batches
    loss_lbl = np.mean(total['loss_lbl'])
    loss_r2a = np.mean(total['loss_r2a'])
    loss_a2r = np.mean(total['loss_a2r'])
    loss_wd  = np.mean(total['loss_wd'])
    loss_src_lm = np.mean(total['loss_src_lm'])
    loss_tar_lm = np.mean(total['loss_tar_lm'])

    loss_total = np.mean( total['loss_lbl'] + args.l_wd * total['loss_wd'] + args.l_r2a *
        total['loss_r2a'] + args.l_a2r * total['loss_a2r'] + args.l_lm * (total['loss_src_lm'] + total['loss_tar_lm']))

    loss_encoder = np.mean(total['loss_lbl'] + args.l_wd * total['loss_wd']
            + args.l_lm * (total['loss_src_lm'] + total['loss_tar_lm']))

    loss_lbl_r2a = np.mean(total['loss_lbl'] + args.l_r2a * total['loss_r2a'] + args.l_a2r *
            total['loss_a2r'])

    print("{:15s} {:s} {:.4f}, {:s} {:.4f}, {:s} {:.4f} * {:.1e}, {:s} {:.4f} * {:.1e} {:s} {:.4f} * {:.1e},"\
            " {:s} {:.4f} * {:.1e}, {:s} {:.4f} * {:.1e}".format(
                task,
                colored("l_tot", "red"),
                loss_total,
                colored("l_lbl", "red"),
                loss_lbl,
                colored("l_wd", "red"),
                loss_wd,
                args.l_wd,
                colored("l_src_lm", "red"),
                loss_src_lm,
                args.l_lm,
                colored("l_tar_lm", "red"),
                loss_tar_lm,
                args.l_lm,
                colored("l_r2a", "red"),
                loss_r2a,
                args.l_r2a,
                colored("l_a2r", "red"),
                loss_a2r,
                args.l_a2r))

    acc, f1, recall, precision = -1, -1, -1, -1
    if args.num_classes[task] > 1:
        acc, f1, recall, precision = _compute_score(y_pred=total['pred_lbl'],
            y_true=total['true_lbl'], num_classes=args.num_classes[task])

        print("{:15s} {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                '',
                colored("acc", "blue"),
                acc,
                colored("recall", "blue"),
                recall,
                colored("precision", "blue"),
                precision,
                colored("f1", "blue"),
                f1
            ))

        print(metrics.confusion_matrix(y_true=total['true_lbl'], y_pred=total['pred_lbl']))

    return {
            'loss_lbl':    loss_lbl,
            'loss_r2a':    loss_r2a,
            'loss_lbl_r2a':loss_lbl_r2a,
            'loss_a2r':    loss_a2r,
            'loss_wd':     loss_wd,
            'loss_src_lm':     loss_src_lm,
            'loss_tar_lm':     loss_tar_lm,
            'loss_encoder':loss_encoder,
            'loss_total':  loss_total,
            'acc':         acc,
            'f1':          f1,
            'recall':      recall,
            'precision':   precision,
            }

def evaluate_r2a_batch(model, task, batch, args, writer=None):
    '''
        Evaluate the network on a batch of examples for the target task

        model: a dictionary of networks
        task: the name of the task
        batch: a batch of examples for the specified task
        args: the overall argument
        writer: a file object. If not none, will write the prediction result and the generated
            attention to the file
    '''
    # get the current batch
    text, rat_freq, rationale, gold_att, _, text_len, label, raw, _ = batch

    # convert to variable and tensor
    text, text_len, rat_freq, rationale, gold_att = _to_tensor(
            [text, text_len, rat_freq, rationale, gold_att], args.cuda)
    text_mask = _get_mask(text_len, args.cuda)

    # Encoder
    hidden, _ = model['encoder'](text, text_len, False)
    invar_hidden = model['transform'](hidden)

    # run r2a to generate attention
    pred_att, log_pred_att = model['r2a'](invar_hidden, rationale, rat_freq, text_len, text_mask)

    normalized_rationale = rationale * text_mask
    normalized_rationale = normalized_rationale / torch.sum(normalized_rationale, dim=1,
            keepdim=True)
    uniform = text_mask / torch.sum(text_mask, dim=1, keepdim=True)

    loss_p2g = 1 - F.cosine_similarity(pred_att, gold_att)
    loss_p2g = _to_numpy(loss_p2g)

    loss_rationale = 1 - F.cosine_similarity(normalized_rationale, gold_att)
    loss_rationale = _to_numpy(loss_rationale)

    loss_uniform = 1 - F.cosine_similarity(uniform, gold_att)
    loss_uniform = _to_numpy(loss_uniform)

    gold_att, rationale, pred_att, rat_freq = _to_numpy([gold_att, rationale, pred_att, rat_freq])

    # Write attention to a tsv file (provide data for R2A model)
    if writer:
        data_utils.write(writer, task, raw, label, gold_att, rationale, pred_att, rat_freq)

    return { 
            'loss_p2g':    loss_p2g,
            'loss_uniform': loss_uniform,
            'loss_rationale': loss_rationale,
            }

def evaluate_r2a(data, task, model, args, writer=None):
    '''
        Only applied to mode=test_r2a. Generate r2a for target data

        data: the data to be tested on
        task: the name of the task
        model: a dictionary of networks
        args: the overall argument
        writer: a file object. If not none, will write the prediction result and the generated
            attention to the file
    '''
    if writer:
        writer.write('task\tlabel\traw\trationale\tpred_att\tgold_att\trat_freq\n')

    for key in model.keys():
        model[key].eval()

    # obtain an iterator to go through the test data
    batches = data_utils.data_loader(data, args.batch_size, shuffle=False)

    total = {}

    # Iterate over the test data. Concatenate all the results.
    for batch in batches:
        cur_res = evaluate_r2a_batch(model, task, batch, args, writer)

        # store results of current batch
        for key, value in cur_res.items():
            if key not in total:
                total[key] = value
            else:
                total[key] = np.concatenate((total[key], value))

    loss_p2g = np.mean(total['loss_p2g'])

    print("{:15s} {:s} {:.4f} {:s} {:.4f} {:s} {:.4f}".format(
        task, 
        colored("l_p2g", "blue"), loss_p2g,
        colored("l_rat", "blue"), np.mean(total['loss_rationale']),
        colored("l_unf", "blue"), np.mean(total['loss_uniform']),
        ))

    return loss_p2g
