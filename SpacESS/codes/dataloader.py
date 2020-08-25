#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import re

from torch.utils.data import Dataset
from sklearn.utils import shuffle as shuffle

def split_body(body, atom):
    '''
        input: body in form "atom" or "atom1, atom2"
        returns: list of all atoms in the body
    '''
    body_sep = ',\s*'
    pattern = re.compile(atom + body_sep + atom)
    groups = re.match(pattern, body)
    if groups:
        return list(groups.groups())
    else: return [body]
    '''try:
        premise1, premise2 = re.match(pattern, body).groups()
        print('successful split: ')
        print('premise 1 ', premise1)
        print('premise 2 ', premise2)
    except:
        return [body]   # body has one atom only
    return [premise1, premise2]'''




def split_atom(atom):
    '''
        takes atom in form of "rel(X,Y)", returns r, X, Y
    '''
    pattern = re.compile('(.*)\((X\d),\s*(X\d)')
    rel, ent1, ent2 = re.match(pattern, atom).groups()
    return rel, ent1, ent2

def read_clauses(filename, relation2id):
    clauses = []
    clentity2id = {}  # entity in a clause name to id
    atom = "(.*\(X\d,\s*X\d\))"
    clause_sep = '\s*\:\-\s*'
    rule_clause = re.compile(atom + clause_sep + atom)
    curr_id = 0
    with open(filename, 'r') as f:
        for num, clause in enumerate(f):
            concl, body = re.match(rule_clause, clause).groups()
            body = split_body(body, atom)  # list of atoms in the body of the clause
            r, h, t = split_atom(concl)
            assert h != t, 'Reflexive relationship '+r
            entity_name = 'cl'+str(num)
            conclusion_h_id = curr_id; conclusion_t_id = curr_id+1
            curr_id += 2
            clentity2id[entity_name+h] = conclusion_h_id
            clentity2id[entity_name+t] = conclusion_t_id
            conclusion_triple = [conclusion_h_id, relation2id[r], conclusion_t_id]
            premise_triples = []
            for premise in body:
                r, h, t = split_atom(premise)
                h_name = entity_name+h
                t_name = entity_name+t
                if h_name not in clentity2id.keys():
                    clentity2id[h_name] = curr_id; curr_id += 1
                if t_name not in clentity2id.keys():
                    clentity2id[t_name] = curr_id; curr_id += 1
                premise_triples.append([clentity2id[h_name], relation2id[r], clentity2id[t_name]])
            cl = Clause(clause, premise_triples, conclusion_triple)
            clauses.append(cl)
    return clauses, clentity2id

class Rule():
    '''
        represents an unlabeled triple
    '''
    def __init__(self, premise, conclusion, conf):
        self.premise = premise
        self.conclusion = conclusion
        self.conf = conf

class Clause:
    '''
        represents clause, as used for the adversarial rule injection
    '''
    def __init__(self, statement, premises, conclusion):
        self.premises = premises   #[head_id, rel_id, tail_id] atom representation
        self.conclusion = conclusion
        self.clause_stmt = statement    # string statement of the clause
        self.is_conjunction = (len(premises) == 2)   # clause has two atoms in the premise



class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.idx = 1


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.from_numpy(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        self.idx = torch.tensor([idx])

        return self.idx, positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        idx = torch.stack([_[0] for _ in data], dim=0)
        positive_sample = torch.stack([_[1] for _ in data], dim=0)
        negative_sample = torch.stack([_[2] for _ in data], dim=0)
        subsample_weight = torch.cat([_[3] for _ in data], dim=0)
        mode = data[0][4]
        return idx, positive_sample, negative_sample, subsample_weight, mode


    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        self.idx = torch.tensor([idx])
        return self.idx, positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        idx  = torch.stack([_[0] for _ in data], dim=0)
        positive_sample = torch.stack([_[1] for _ in data], dim=0)
        negative_sample = torch.stack([_[2] for _ in data], dim=0)
        filter_bias = torch.stack([_[3] for _ in data], dim=0)
        mode = data[0][4]
        return idx, positive_sample, negative_sample, filter_bias, mode

class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
