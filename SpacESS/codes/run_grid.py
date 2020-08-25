#usr/bin/python3

####
# TODO:
#     1) Make sure this works on parallel as well!
###

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

GAMMA1 = [1. ]
GAMMA2 = [1.1]
N_NEGS_LIST = [50, 100]
N_STEPS_LIST = [150000, 200000]
OPT = 'adam'

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--parallel', action='store_true', help='parallelize over several gpus')
    parser.add_argument('--loss', default = 'custom')

    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('--train_old', action = 'store_true', help = 'Use original train step function')

    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None,
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('-g1', '--gamma1', type=float)
    parser.add_argument('--diff', type=float, default=.1)

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(args.save_path or args.init_checkpoint, args.model + '_'+ OPT + '_grid.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def grid_train_model(init_step, valid_triples, all_true_triples, kge_model, train_iterator, args):

    if args.warm_up_steps:
        warm_up_steps = args.warm_up_steps
    else:
        warm_up_steps = args.max_steps // 2

    current_learning_rate = args.learning_rate
    if OPT == 'adam': opt_fnc = torch.optim.Adam
    elif OPT == 'ada': opt_fnc = torch.optim.Adagrad
    else: print("Unknown optimizer")

    optimizer = opt_fnc(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate )

    train_fnc = kge_model.train_step if not args.train_old else kge_model.train_step_old
    training_logs = []

    #Training Loop
    for step in range(init_step, args.max_steps):
        log = train_fnc(kge_model, optimizer, train_iterator, args)
        training_logs.append(log)

        if step >= warm_up_steps:
            current_learning_rate = current_learning_rate / 10
            logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
            optimizer = opt_fnc(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate
            )
            warm_up_steps = warm_up_steps * 3

        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
            log_metrics('Training average', step, metrics)
            training_logs = []

        if args.do_valid and step % args.valid_steps == 0:
            logging.info('Evaluating on Valid Dataset...')
            model_module = kge_model.module if args.parallel else kge_model
            metrics = model_module.test_step(model_module, valid_triples, all_true_triples, args)
            print('Eval on valid dataset at step ', step)
            for k, val in metrics.items():
                print(k, ": ", val)
            log_metrics('Valid', step, metrics)

    save_variable_list = {
        'step': step,
        'current_learning_rate': current_learning_rate,
        'warm_up_steps': warm_up_steps
    }
    return step

def construct_dataloader(args, train_triples, nentity, nrelation):
    train_dataloader_head = DataLoader(
        TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, args.cpu_num//2),
        collate_fn=TrainDataset.collate_fn
    )

    train_dataloader_tail = DataLoader(
        TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, args.cpu_num//2),
        collate_fn=TrainDataset.collate_fn
    )

    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

    return train_iterator


def main(args):

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions


    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))

    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    current_learning_rate = args.learning_rate

    ntriples = len(train_triples)

    '''print('Model: %s' % args.model)
                print('Data Path: %s' % args.data_path)
                print('#entity: %d' % nentity)
                print('#relation: %d' % nrelation)
                print('optimizer: ', OPT)
                if args.train_old: print('USING ORIGINAL TRAINING FUNCTION')

                print('learning_rate = %f' % current_learning_rate)
                print('batch_size = %d' % args.batch_size)
                print('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
                print('hidden_dim = %d' % args.hidden_dim)
                print('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
                if args.negative_adversarial_sampling:
                    print('adversarial_temperature = %f' % args.adversarial_temperature)
    '''

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('optimizer: %s' % OPT)
    if args.train_old: logging.info('USING ORIGINAL TRAINING FUNCTION')
    #else: print('GRID TESTING\nUsing new loss function')

    info = 'Model - {}; opt - {}; batch size - {}, dim - {}; dataset - {}; lr - {}; '.format(
        args.model, OPT, str(args.batch_size), args.hidden_dim, args.data_path, str(current_learning_rate))
    print(info)
    for g1, g2 in zip(GAMMA1, GAMMA2):
        for n_neg in N_NEGS_LIST:
            for steps in N_STEPS_LIST:
                current_learning_rate = args.learning_rate
                # re-initialize the model
                kge_model = KGEModel(
                    model_name=args.model,
                    nentity=nentity,
                    nrelation=nrelation,
                    ntriples = ntriples,
                    hidden_dim=args.hidden_dim,
                    gamma=args.gamma,
                    gamma1 = g1,
                    gamma2 = g2,
                    double_entity_embedding=args.double_entity_embedding,
                    double_relation_embedding=args.double_relation_embedding,
                )
                kge_model.set_loss(args.loss)

                logging.info('Model Parameter Configuration:')
                for name, param in kge_model.named_parameters():
                    logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
                logging.info('Loss function %s' % args.loss)
                if args.cuda:
                    kge_model = kge_model.cuda()

                logging.info('Ramdomly Initializing %s Model...' % args.model)

                args.max_steps = steps
                args.negative_sample_size = n_neg
                out_line = 'g1 = {}, g2 = {}, #steps = {}, #negs = {};'.format(kge_model.gamma1, kge_model.gamma2, args.max_steps, args.negative_sample_size)
                logging.info('gamma1 = %f, gamma2 = %f' % (g1, g2))
                logging.info('Max steps - %d' % args.max_steps)
                logging.info('Negative sample %d ' % args.negative_sample_size)

                train_iterator = construct_dataloader(args, train_triples, nentity, nrelation)
                step = grid_train_model(0, valid_triples, all_true_triples, kge_model, train_iterator, args)
                metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
                log_metrics('Test', step, metrics)
                values = [str(metrics['MRR']), str(metrics['MR']), str(metrics['HITS@1']), str(metrics['HITS@3']), str(metrics['HITS@10'])]
                out_line = out_line + ';'.join(values)
                print(out_line)

                logging.info('\n-----------------------------------------------')





if __name__ == '__main__':
    main(parse_args())

