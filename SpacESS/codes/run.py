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
import itertools

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel
from adversarial import ADVModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
from dataloader import Rule
import dataloader as dt


#LOSS_TMP = 'custom'
OPT_STOPPING = False
STEPS_BEFORE_VALID = 0   # number of steps to do before running on validation set
RULE_BATCH_SIZE_INV = 1000
RULE_BATCH_SIZE_IMPL = 1000
RULE_BATCH_SIZE_EQ = -1
RULE_BATCH_SIZE_SYM = -1
GAMMA1 = [8]
GAMMA2 = [10]
N_NEGS_LIST = [10] #Number of negative samples
N_STEPS_LIST = [100000]

# rules settings
RULE_TYPES = []
EPSILONS_INV = [.0001]
EPSILONS_IMPL = [.0001]
EPSILONS_SYM = [0]
EPSILONS_EQ = [0]

WEIGHTS_INV = [2]
WEIGHTS_IMPL = [1]
WEIGHTS_SYM = [.0005]
WEIGHTS_EQ = [.001]

LOSSES = []
MODELS = []
DIMENSIONS = []

TEST_SYMMETRY = False

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--project', action = 'store_true')
    parser.add_argument('--ruge', action = 'store_true', help = 'Use RUGE rules')
    parser.add_argument('--inject', action = 'store_true', help = 'Old inject option') 
    parser.add_argument('--inject_mine', action ='store_true', help = 'Inject rules using my model')
    parser.add_argument('--ruge-inject', action = 'store_true', help = 'Inject rules using RUGE model')
    parser.add_argument('--adversarial', action = 'store_true', help = 'Use adversarial rule injection')
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--parallel', action='store_true', help='parallelize over several gpus')
    parser.add_argument('--loss', default = 'rotate')
    #parser.add_argument('--rules', action = 'store_true')
    parser.add_argument('--eq', action = 'store_true', help = 'use equality rules in training')
    parser.add_argument('--inv', action = 'store_true', help = 'use inverse rules in training')
    parser.add_argument('--impl', action = 'store_true', help = 'use implication rules in training')
    parser.add_argument('--sym', action = 'store_true', help = 'use symmetry rules in training')

    parser.add_argument('--do_experiment', action = 'store_true', help = 'Use updated loss function')
    parser.add_argument('--do_grid', action = 'store_true', help = 'Grid testing')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

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
    parser.add_argument('--l2-r', default = 0.0, type = float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('-g1', '--gamma1', type=float)
    parser.add_argument('--diff', type=float, default=.1)
    parser.add_argument('--opt', default='ada')

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

    args = parser.parse_args(args)

    return args

def cycle(iterable):
    ''' helper for rule iterators '''
    while True:
        for x in iterable:
            yield x

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

def save_model(model, optimizer, save_variable_list, args, idx):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config'+idx+'.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint'+idx)
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'+idx),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding' + idx),
        relation_embedding
    )

    if args.model == 'SpacESS':
        relation_embedding_head = model.rotator_head.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'relation_embedding_head' + idx),
            relation_embedding_head
        )

        relation_embedding_tail = model.rotator_tail.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'relation_embedding_tail' + idx),
            relation_embedding_tail
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

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train_'+args.model+'.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test_'+args.model+'.log')

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

def setup_rule_loader(n_batches, batch_size, dir_path, filename, device, rule_batchsize  = 0):
    rules = np.loadtxt(os.path.join(dir_path, filename))
    rules = torch.LongTensor(rules).to(device)
    #batch_size = rules.shape[0]//n_batches

    batch_size = rules.shape[0]//n_batches
    if batch_size == 0:
        batch_size = rules.shape[0]     # reset batch to include all rules if too few rules
    batch_size = rule_batchsize
    if batch_size == -1 or rule_batchsize == -1:
        batch_size = rules.shape[0]//n_batches
        if batch_size == 0:
            batch_size = rules.shape[0]     # reset batch to include all rules if too few rules

    dl = DataLoader(rules, batch_size = int(batch_size), shuffle = True, drop_last = False)
    return rules.shape[0], batch_size, iter(cycle(dl))

def train_model(init_step, valid_triples, all_true_triples, kge_model, adv_model, train_iterator, rule_iterators, args, idx = ''):
    if args.warm_up_steps:
        warm_up_steps = args.warm_up_steps
    else:
        warm_up_steps = args.max_steps // 2   # 2

    current_learning_rate = args.learning_rate
    if args.opt == 'adam': optim_fnc = torch.optim.Adam
    elif args.opt == 'ada': optim_fnc = torch.optim.Adagrad
    else: raise ValueError('optimizer %s not supported '% args.opt)

    # NOTE: added weight decay (L2 regularization) to optim function - test on SpacESS!
    optimizer = optim_fnc(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr = current_learning_rate)

    if args.adversarial:
        adv_optimizer = optim_fnc(
                filter(lambda p: p.requires_grad, adv_model.parameters()),
                lr = 1*current_learning_rate)

    training_logs = []
    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        current_learning_rate = checkpoint['current_learning_rate']
        warm_up_steps = checkpoint['warm_up_steps']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    #train_step = kge_model.ruge_train_step if args.ruge else kge_model.train_step
    for step in range(init_step, args.max_steps):
        # get batches of rules
        rule_batches = {}
        for key, iterator in rule_iterators.items():
            rule_batches[key] = next(iterator)

        # adv step
        if args.adversarial:
            errors = adv_model.train_adversarial(adv_model, kge_model, adv_optimizer)
        log = kge_model.train_step(kge_model, adv_model, optimizer, train_iterator, args, rules = rule_batches)

        training_logs.append(log)

        if step >= warm_up_steps and not OPT_STOPPING:
            current_learning_rate = current_learning_rate / 10
            logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
            optimizer = optim_fnc(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate
            )
            warm_up_steps = warm_up_steps * 2

        if step % args.save_checkpoint_steps == 0 and step > 0:
            save_variable_list = {
                'step': step,
                'current_learning_rate': current_learning_rate,
                'warm_up_steps': warm_up_steps
                }
            save_model(kge_model, optimizer, save_variable_list, args, idx)

        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
            log_metrics('Training average', step, metrics)
            if args.adversarial:
                logging.info('# adversarial errors - ' + ', '.join([str(x) for x in errors]))
                #logging.info('N adversarial errors per adv epoch: {}, {}'.format(errors[0], errors[-1]))
            training_logs = []

        if args.do_valid and step % args.valid_steps == 0 and step >= STEPS_BEFORE_VALID:
            
            logging.info('Evaluating on Valid Dataset...')
            model_module = kge_model.module if args.parallel else kge_model
            metrics = model_module.test_step(model_module, valid_triples, all_true_triples, args)
            log_metrics('Valid', step, metrics)
            if args.do_grid:
                info = 'Validation (%d): ' % step
                for key, val in metrics.items():
                    info = info + key + ' - ' + str(val) + ';'
                print(info)

    save_variable_list = {
        'step': step,
        'current_learning_rate': current_learning_rate,
        'warm_up_steps': warm_up_steps
    }
    save_model(kge_model, optimizer, save_variable_list, args, idx)
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

def reset_empty_values(args):
    global DIMENSIONS
    global LOSSES
    global MODELS
    global GAMMA1, GAMMA2
    global RULE_TYPES
    global WEIGHTS_INV, WEIGHTS_IMPL, WEIGHTS_SYM, WEIGHTS_EQ
    global EPSILONS_INV, EPSILONS_IMPL, EPSILONS_SYM, EPSILONS_EQ

    RULE_TYPES = np.array(['inverse', 'implication', 'symmetry', 'equality'])[[args.inv, args.impl, args.sym, args.eq]]
    if not args.inv:
        EPSILONS_INV, WEIGHTS_INV = [0], [0]
    if not args.impl:
        EPSILONS_IMPL, WEIGHTS_IMPL = [0], [0]
    if not args.sym:
        EPSILONS_SYM, WEIGHTS_SYM = [0], [0]
    if not args.eq:
        EPSILONS_EQ, WEIGHTS_EQ = [0], [0]

    if len(DIMENSIONS) == 0:
        DIMENSIONS = [args.hidden_dim]
    if len(MODELS) == 0:
        MODELS = [args.model]
    if len(LOSSES) == 0:
        LOSSES = [args.loss]
    if args.loss != 'custom':
        GAMMA1 = [0]; GAMMA2 = [0]

def print_rules_info(model, args):
    if not (args.inv or args.impl or args.eq or args.sym):
        return
    weight_info = 'Weights: '
    eps_info = 'Epsilons: '
    if args.inv:
        weight_info += 'inv - ' + str(model.rule_weight['inverse']) + ';'
        eps_info += 'inv - ' + str(model.epsilon_inv) + ';'
    if args.impl:
        weight_info += 'impl - '+str(model.rule_weight['implication']) + ';'
        eps_info += 'impl - ' + str(model.epsilon_impl) + ';'
    if args.sym:
        weight_info += 'sym - '+str(model.rule_weight['symmetry']) + ';'
        eps_info += 'sym - ' + str(model.epsilon_sym) + ';'
    if args.eq:
        weight_info += 'eq - '+str(model.rule_weight['equality']) + ';'
        eps_info += 'eq - ' + str(model.epsilon_eq) + ';'

    print(weight_info)
    print(eps_info)


def run_grid(nentity, nrelation, train_triples,
            valid_triples, test_triples, all_true_triples, args, rule_iterators = None, adv_model = None): 
    ntriples = len(train_triples)

    if args.inject:
        print('injecting rules')
    else:
        print('rules not injected')

    if args.ruge:
        print('Using RUGE injection model')

    reset_empty_values(args)
    current_learning_rate = args.learning_rate

    if args.negative_adversarial_sampling:
        print('Temperature - ', args.adversarial_temperature); print()

    info = 'Model - {}; opt - {}; batch size - {}; dataset - {}; lr - {}, gamma = {}; '.format(
        args.model, args.opt, args.batch_size, args.data_path, current_learning_rate, args.gamma)
    info2 = 'Loss fnc - {}; inv - {}; impl - {}; sym - {}; eq - {}'.format(
        args.loss, args.inv, args.impl, args.sym, args.eq)
    print(info)
    print(info2)

    current_learning_rate = args.learning_rate
    EPSILONS = itertools.product(EPSILONS_INV, EPSILONS_IMPL, EPSILONS_SYM, EPSILONS_EQ)
    WEIGHTS = itertools.product(WEIGHTS_INV, WEIGHTS_IMPL, WEIGHTS_SYM, WEIGHTS_EQ)

    idx = -1 # for saving models with several parameters
    for g1, g2 in zip(GAMMA1, GAMMA2):
        for eps_inv, eps_impl, eps_sym, eps_eq in EPSILONS:
            for w_inv, w_impl, w_sym, w_eq in WEIGHTS:
                for dim, n_negs, steps in itertools.product(DIMENSIONS, N_NEGS_LIST, N_STEPS_LIST):
                    idx += 1
                    # re-initialize the model
                    kge_model = KGEModel(
                        model_name=args.model,
                        nentity=nentity,
                        nrelation=nrelation,
                        ntriples = ntriples,
                        hidden_dim=dim,
                        args = args
                    )
                    if 'inverse' in RULE_TYPES:
                        kge_model.rule_weight['inverse'] = w_inv
                        kge_model.epsilon_inv = eps_inv
                    if 'implication' in RULE_TYPES:
                        kge_model.rule_weight['implication'] = w_impl
                        kge_model.epsilon_impl = eps_impl
                    if 'symmetry' in RULE_TYPES:
                        kge_model.rule_weight['symmetry'] = w_sym
                        kge_model.epsilon_sym = eps_sym
                    if 'equality' in RULE_TYPES:
                        kge_model.rule_weight['equality'] = w_eq
                        kge_model.epsilon_eq = eps_eq

                    kge_model.set_loss(args.loss)
                    logging.info('Model: %s' % args.model)
                    logging.info('Data Path: %s' % args.data_path)
                    logging.info('#entity: %d' % nentity)
                    logging.info('#relation: %d' % nrelation)
                    logging.info('optimizer: %s' % args.opt)
                    logging.info('learning rate: %f' % current_learning_rate)
                    logging.info('loss: %s' % args.loss)
                    if args.inv:
                        logging.info('using inverse rules: eps = %f, weight = %f' % (kge_model.epsilon_inv, kge_model.rule_weight['inverse']))
                    if args.impl:
                        logging.info('using implication rules: eps = %f, weight = %f' % (kge_model.epsilon_impl, kge_model.rule_weight['implication']))
                    if args.sym:
                        logging.info('using symmetry rules: eps = %f, weight = %f' % (kge_model.epsilon_sym, kge_model.rule_weight['symmetry']))
                    if args.eq:
                        logging.info('using equality rules: eps = %f, weight = %f' % (kge_model.epsilon_eq, kge_model.rule_weight['equality']))
                    logging.info('Model Parameter Configuration:')
                    for name, param in kge_model.named_parameters():
                        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
                    logging.info('Loss function %s' % args.loss)
                    if args.cuda:
                        kge_model = kge_model.cuda()

                    logging.info('Ramdomly Initializing %s Model...' % args.model)

                    print_rules_info(kge_model, args)
                    args.max_steps = steps
                    args.negative_sample_size = n_negs
                    #out_line = '#steps = {}, #negs = {};'.format(args.max_steps, args.negative_sample_size)
                    logging.info('Max steps - %d' % args.max_steps)
                    logging.info('Negative sample %d ' % args.negative_sample_size)
                    assert kge_model.inject == args.inject, 'Inject is wrong'
                    # train
                    train_iterator = construct_dataloader(args, train_triples, nentity, nrelation)
                    step = train_model(0, valid_triples, all_true_triples, kge_model, adv_model, train_iterator, rule_iterators, args, str(idx))

                    # valid
                    logging.info('Evaluating on Valid Dataset...')
                    metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                    #metrics1 = kge_model.getScore(kge_model, valid_triples, all_true_triples, args)
                    log_metrics('Valid', step, metrics)
                    info = 'Validation (%d): ' % step
                    for key, val in metrics.items():
                        info = info + key + ' - ' + str(val) + ';'
                    print(info)
                    # test
                    out_line = '#steps = {}, #negs = {}, dim = {};'.format(step, args.negative_sample_size, kge_model.hidden_dim)
                    metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
                    print("Hello")
                    metrics1 = kge_model.getScore(kge_model, test_triples, all_true_triples, args)
                    log_metrics('Test', step, metrics)
                    values = [str(metrics['MRR']), str(metrics['MR']), str(metrics['HITS@1']), str(metrics['HITS@3']), str(metrics['HITS@10'])]
                    out_line = out_line + ';'.join(values)
                    print(out_line)

                    logging.info('\n-----------------------------------------------')
            print()


def read_ruleset_ruge(data_path, rule_type, premise_idx, concl_idx):
    fname = 'groundings_'+rule_type+'_confidence.txt'
    rules_info = np.loadtxt(os.path.join(data_path, fname))

    premise = rules_info[:, premise_idx]
    conclusion = rules_info[:, concl_idx]
    conf = rules_info[:, -1]
    rules = [Rule(premise = p, conclusion = c, conf = cf) for p, c, cf in zip(premise, conclusion, conf)]
    return np.array(rules)


def construct_ruge_loader(n_batches, args):
    ds = args.data_path.split('/')[-1]
    if ds == 'FB15k':
        rules = read_ruleset_ruge(args.data_path, "inverse", [0, 2, 1], [1, 3, 0])
        rules = np.append(rules, read_ruleset_ruge(
            args.data_path, "implication", [0, 2, 1], [0, 3, 1] ))
        rules = np.append(rules, read_ruleset_ruge(
            args.data_path, "equality", [0, 2, 1], [0, 3, 1]))
        rules = np.append(rules, read_ruleset_ruge(
            args.data_path, "symmetric", [0, 2, 1], [1, 2, 0]))
    elif ds == 'FB15k-237':
        rules = read_ruleset_ruge(args.data_path, "inverse", [0, 2, 1], [1, 3, 0])
        rules = np.append(rules, read_ruleset_ruge(
            args.data_path, "implication", [0, 2, 1], [0, 3, 1] ))
        rules = np.append(rules, read_ruleset_ruge(
            args.data_path, "equality", [0, 2, 1], [0, 3, 1]))
    elif ds == 'wn18':
        rules = read_ruleset_ruge(args.data_path, 'inverse', [0, 2, 1], [1, 3, 0])
    else:
        raise ValueError('Dataset %s does not have rules!' % ds)

    ruge_loader = generator(rules, int(n_batches))
    return len(rules), ruge_loader

def generator(rules, n_batches):
    '''
    Generates batches; used for RUGE rule loading
    '''
    np.random.shuffle(rules)
    batches = np.array_split(rules, n_batches)
    curr_batch = 0
    while True:
        if curr_batch == 0:
            np.random.shuffle(rules)
            batches = np.array_split(rules, n_batches)
        yield batches[curr_batch]
        curr_batch = (curr_batch + 1) % n_batches


def main(args):
    if not torch.cuda.is_available():
        args.cuda = False

    if args.ruge:
        args.loss = 'ruge'

    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and (not args.do_experiment) and (not args.do_grid):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)

    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console

    set_logger(args)
    if args.regularization != 0:
        print('L3 regularization with coeff - ', args.regularization)
    if args.l2_r != 0:
        print('L2 regularization with coeff - ', args.l2_r)
    if args.project != 0:
        print('projecting before training')
    #logging.info('Inverse loss = premise - concl (reverse)')
    if OPT_STOPPING:
        logging.info('Opt stopping is ON')
        print('Opt stopping is on')


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

    if args.inject:
        logging.info('With rule injection')
    else:
        logging.info('NO INJECTION')

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id) # For testing on Symmetric in WordNet: Symmetric_testWN18
    logging.info('#test: %d' % len(test_triples))

    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    train_args = {}

    # set up rule iterators
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_batches = len(train_triples) // args.batch_size
    if n_batches < len(train_triples) / args.batch_size: n_batches += 1
    rule_iterators = {}
    rules_info = ''
    if args.inv:
        n_inverse, inverse_batchsize, rule_iterators['inverse'] = setup_rule_loader(n_batches, args.batch_size, args.data_path, 'groundings_inverse.txt', device, RULE_BATCH_SIZE_INV)
        rules_info += 'Inverse: batch size %d out of %d rules' %( inverse_batchsize, n_inverse) + '\n'
    if args.eq:
        n_eq, eq_batchsize, rule_iterators['equality'] = setup_rule_loader(n_batches, args.batch_size, args.data_path, 'groundings_equality.txt', device, RULE_BATCH_SIZE_EQ)
        rules_info += 'Equality: batch size %d out of %d rules' % (eq_batchsize, n_eq) + '\n'
    if args.impl:
        n_impl, impl_batchsize, rule_iterators['implication'] = setup_rule_loader(n_batches, args.batch_size, args.data_path, 'groundings_implication.txt', device, RULE_BATCH_SIZE_IMPL)
        rules_info += 'implication: batch size %d out of %d rules\n' % (impl_batchsize, n_impl)
    if args.sym:
        n_symmetry, sym_batchsize, rule_iterators['symmetry'] = setup_rule_loader(n_batches, args.batch_size, args.data_path, 'groundings_symmetric.txt', device, RULE_BATCH_SIZE_SYM)
        rules_info += 'symmetry: batch size %d out of %d rules\n' % (sym_batchsize, n_symmetry)
    if args.ruge or args.ruge_inject:
        n_rules, rule_iterators['ruge'] = construct_ruge_loader(n_batches, args)
        rules_info += 'RUGE: Total %d rules\n' % n_rules


    if rules_info:
        logging.info(rules_info)


    # ----------- adversarial ------------------
    if args.adversarial:
        clauses_filename = os.path.join(args.data_path, 'clauses_0.9.pl')
        adv_clauses, clentity2id = dt.read_clauses(clauses_filename, relation2id)
        n_clause_entities = len(clentity2id)
        mult = 2
        if args.model in ['TransE', 'pRotatE']: mult = 1
        if 'QuatE' in args.model: mult = 4
        adv_model = ADVModel(
            clauses = adv_clauses,
            n_entities = len(clentity2id) ,
            dim = mult*args.hidden_dim,
            use_cuda = args.cuda
        )
        if args.cuda:
            adv_model = adv_model.cuda()
    else:
        adv_model = None

    if args.do_grid:
        if rules_info:
            print(rules_info)
        run_grid(nentity, nrelation, train_triples, valid_triples, test_triples, all_true_triples, args, rule_iterators, adv_model)
        exit()
    ntriples = len(train_triples)
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        ntriples = ntriples,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
    )
    kge_model.set_loss(args.loss)

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    logging.info('Loss function %s' % args.loss)
    if args.cuda and args.parallel:
        gpus = [0, 1]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpus)
        kge_model.cuda()
        kge_model = torch.nn.DataParallel(kge_model, device_ids = [0, 1])

    elif args.cuda:
        kge_model = kge_model.cuda()

    if args.do_train or args.do_experiment:
        # Set training dataloader iterator
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

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )


        
    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logging.info('Ramdomly Initializing %s Model...' % args.model)
            init_step = 0

    step = init_step

    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    # Set valid dataloader as it would be evaluated during training



    if args.do_train:
        train_model(init_step, valid_triples, all_true_triples, kge_model, train_iterator, len(train_triples), args)


    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        model_module = kge_model.module if args.parallel else kge_model
        metrics = model_module.test_step(kge_model, train_triples, all_true_triples, args)
        #metrics1 = model_module.getScore(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

    # experiment on the updated function
    if args.do_experiment:
        logging.info('\n\nSTARTING EXPERIMENT\n')

    train_model(init_step, valid_triples, all_true_triples, kge_model, train_iterator, rule_iterators, args)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        model_module = kge_model.module if args.parallel else kge_model
        metrics = model_module.test_step(kge_model, valid_triples, all_true_triples, args)
        #metrics1 = model_module.getScore(kge_model, train_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        model_module = kge_model.module if args.parallel else kge_model
        metrics = model_module.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        model_module = kge_model.module if args.parallel else kge_model
        metrics = model_module.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

if __name__ == '__main__':
    main(parse_args())


