''' Adversarial training '''

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

dataset = 'wn'

class ADVModel(nn.Module):
    def __init__(self, clauses, n_entities, dim, use_cuda):
        super(ADVModel, self).__init__()
        '''
            For every clause:
                for every unique atom in the clause:
                    embedding of the entity
            relation embeddings - inherited from the model

        '''
        self.clauses = clauses
        self.use_cuda = use_cuda
        self.clause_entity_embedding = nn.Parameter(torch.zeros((n_entities, dim)))
        nn.init.uniform_(
            tensor=self.clause_entity_embedding,
            a=1,
            b=0
            )
        self.construct_conclusions_data()

        self.construct_premise_data()

        self.adv_epochs = 20
        self.unit_cube = True
        print('Using adversarial with {} epochs'.format(self.adv_epochs))

    def construct_conclusions_data(self):
        '''
            construct head, tail and relations indices of conclusion atoms
        '''
        longTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        self.concl_heads = longTensor([clause.conclusion[0] for clause in self.clauses])
        self.concl_tails = longTensor([clause.conclusion[2] for clause in self.clauses])
        self.concl_rel = longTensor([clause.conclusion[1] for clause in self.clauses])



    def construct_premise_data(self):
        longTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor

        # indices of conjunction clauses
        self.conjunctions = longTensor(
            [i for i, clause in enumerate(self.clauses) if clause.is_conjunction])
        # indices of clauses with one premise only
        self.singles = longTensor([
            i for i, clause in enumerate(self.clauses) if not clause.is_conjunction])

        if self.singles.shape[0] != 0:
            premises = [clause.premises[0] for clause in self.clauses if not clause.is_conjunction]
            self.premise_heads = longTensor([atom[0] for atom in premises])
            self.premise_tails = longTensor([atom[2] for atom in premises])
            self.premise_rel   = longTensor([atom[1] for atom in premises])

        if self.conjunctions.shape[0] != 0:
            conjunction_premises = [clause.premises for clause in self.clauses if clause.is_conjunction]

            self.conj_premise_heads1 = longTensor([atom[0][0] for atom in conjunction_premises])
            self.conj_premise_tails1 = longTensor([atom[0][2] for atom in conjunction_premises])
            self.conj_premise_rel1   = longTensor([atom[0][1] for atom in conjunction_premises])

            self.conj_premise_heads2 = longTensor([atom[1][0] for atom in conjunction_premises])
            self.conj_premise_tails2 = longTensor([atom[1][2] for atom in conjunction_premises])
            self.conj_premise_rel2   = longTensor([atom[1][1] for atom in conjunction_premises])



    def atom_score(self, indices_heads, indices_tails, indices_relations, kge_model, detach):
        heads = torch.index_select(
                                self.clause_entity_embedding,
                                dim=0,
                                index=indices_heads)
        tails = torch.index_select(
                            self.clause_entity_embedding,
                            dim = 0,
                            index = indices_tails)

        heads = heads.unsqueeze(1); tails = tails.unsqueeze(1)
        relations_dict = kge_model.select_relations(indices_relations)
        if detach:
            for key in relations_dict.keys():
                relations_dict[key] = relations_dict[key].detach()

        if not detach:
            heads = heads.detach()
            tails = tails.detach()

        arg_dict = {
            'head': heads,
            **relations_dict,
            'tail': tails
        }
        if kge_model.model_name not in ['biRotatE', 'TransRotatE', 'TransQuatE', 'QuatE', 'sTransQuatE', 'sTransRotatE']:
            arg_dict['mode'] = 'single'

        score = kge_model.compute_score(arg_dict).squeeze()

        return score


    def forward(self, kge_model, detach = True):
        concl_scores = self.atom_score(
                self.concl_heads, self.concl_tails, self.concl_rel, kge_model, detach)
        return_values = [concl_scores]

        # compute premise scores
        if self.singles.shape[0] != 0: # has one atom premises
            premise_scores = self.atom_score(
                    self.premise_heads, self.premise_tails, self.premise_rel, kge_model, detach).squeeze()
            return_values.append(premise_scores)
        else: return_values.append([])

        if self.conjunctions.shape[0] != 0:  # has premises that are conjunction of two atoms
            premise1_scores = self.atom_score(self.conj_premise_heads1,
                    self.conj_premise_tails1, self.conj_premise_rel1, kge_model, detach).squeeze()
            premise2_scores = self.atom_score(self.conj_premise_heads2,
                    self.conj_premise_tails2, self.conj_premise_rel2, kge_model, detach).squeeze()

            conjuction_scores = torch.min(premise1_scores, premise2_scores)
            return_values.append(conjuction_scores)
        else: return_values.append([])
        return return_values


    def project(self):
        with torch.no_grad():
            if self.unit_cube:
                self.clause_entity_embedding.data = self.clause_entity_embedding.clamp(0,1)
            else: # unit sphere
                norms = self.clause_entity_embedding.norm(dim = 1)
                for i in range(self.clause_entity_embedding.shape[0]):
                    self.clause_entity_embedding[i] = \
                        self.clause_entity_embedding[i]/norms[i]


    def _n_errors(self, concl, premise, conj):
        errors = (premise > concl[[self.singles]]).sum()
        if self.conjunctions.shape[0] != 0:
            errors += (conj > concl[[self.conjunctions]]).sum()
        return errors.item()

    def adversarial_loss(self, concl_scores, premise_scores, conj_scores, reverse = False):
        reverse = False
        if not reverse:
            loss = F.relu(-premise_scores + concl_scores[[self.singles]]).mean() # works for fb
        else:
            loss = F.relu(premise_scores - concl_scores[[self.singles]]).mean()
        if self.conjunctions.shape[0] != 0:
            if not reverse:
                loss_conj = F.relu(-conj_scores + concl_scores[[self.conjunctions]]).mean()
            else:
                loss_conj = F.relu(conj_scores - concl_scores[[self.conjunctions]]).mean()
            loss = loss + loss_conj
            loss = loss/2
        #loss = loss/(self.singles.shape[0] + self.conjunctions.shape[0])
        return loss


    @staticmethod
    def train_adversarial(adv_model, kge_model, optimizer):
        log_errors = []
        nn.init.uniform_(
            tensor=adv_model.clause_entity_embedding,
            a=1,
            b=0
            )
        adv_model.train()

        for epoch in range(adv_model.adv_epochs):
            adv_model.project()  # project to unit cube or unit sphere

            optimizer.zero_grad()
            concl_scores, premise_scores, conj_scores = adv_model(kge_model)
            n_errors = adv_model._n_errors(concl_scores, premise_scores, conj_scores)
            loss = adv_model.adversarial_loss(concl_scores, premise_scores, conj_scores)
            loss = loss

            #print('Loss at epoch {} is {}'.format(epoch, loss.item()))

            loss.backward()
            nans = torch.isnan(adv_model.clause_entity_embedding.grad).sum().item()
            if nans != 0:
                print(nans, ' nans detected'); exit()
            optimizer.step()
            log_errors.append(n_errors)

        return log_errors

