import os
import json
from typing import NamedTuple
from tqdm import tqdm

import torch
import torch.nn as nn
from optuna.samplers import TPESampler, GridSampler
import numpy as np
import torch.nn as nn
import torch
import time
import torch.optim as optim
import utils
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import model
import tqdm
import optuna
import dataset
from torch.utils.data import Dataset, DataLoader



class Config():
    """ Hyperparameters for training """
    seed: int = 391275 # random seed
    batch_size: int = 32
    lr: int = 1e-4 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    is_dibert: bool = True

def train_config_to_dict(Config):
        #tokens = [token.to_dict() for token in self.tokens]
    return {'seed': Config.seed,
            'batch_size': Config.batch_size,
            'lr': Config.lr,
            'n_epochs': Config.n_epochs,
            'warmup': Config.warmup, 'is_dibert': Config.is_dibert}


def training(model_dibert, optim,schedular, criterion_cls,criterion_mlm,criterion_pp, train_iter, epoch):

    model_dibert.train()
    losses_cls = []
    losses_mlm = []
    losses_pp = []

    l_cls = []
    p_cls = []
    l_pp = []
    p_pp = []
    l_mlm = []
    p_mlm = []

    softmax = nn.Softmax(dim = -1)
    print('\nTrain_Epoch:', epoch)
    for batch in tqdm.tqdm(train_iter):
        optim.zero_grad()

        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        token_type_ids = batch['token_type_ids'].cuda()
        truelabel_pp = batch['parent_ids'].cuda()
        truelabel_mlm = batch['mask_ids'].cuda()
        truelabel_cls = batch['cls_label'].cuda()


        logits_cls, logits_mlm, logits_pp = model_dibert(input_ids, attention_mask, token_type_ids)
        ## if out dim is (bs x seqlen x numclass) -> (total_words_batch x numclass)
        ## if true label is (bs x seqlen) -> (total_words_batch)
        loss_mlm = criterion_mlm( logits_mlm.view(-1, model.Config.vocab_size), truelabel_mlm.view(-1) )
        loss_cls = criterion_cls (logits_cls.view(-1, 2), truelabel_cls.view(-1 ))

        if(Config.is_dibert==False):
            loss = loss_mlm + loss_cls
        else:
            #loss_pp = criterion_pp(logits_pp.view(-1, model.Config.max_len), truelabel_pp.view(-1) )
            loss_pp = criterion_pp(logits_pp.view(-1, model.Config.vocab_size), truelabel_pp.view(-1))
            loss = loss_mlm + loss_cls + loss_pp
            losses_pp.append(loss_pp.item())

            pred_pp = softmax(logits_pp).argmax(2)
            nptrue_pp, nppreds_pp = utils.prune_preds(truelabel_pp.view(-1), pred_pp.view(-1))
            l_pp.extend(nptrue_pp)
            p_pp.extend(nppreds_pp)


        losses_cls.append(loss_cls.item())
        losses_mlm.append(loss_mlm.item())


        #for now we are only interested in accuracy and f1 of the classification task
        l_cls.extend(truelabel_cls.cpu().detach().numpy())
        preds_cls = softmax(logits_cls).argmax(1)
        p_cls.extend(preds_cls.view(-1).cpu().detach().numpy())


        pred_mlm = softmax(logits_mlm).argmax(2)
        nptrue_mlm, nppreds_mlm = utils.prune_preds(truelabel_mlm.view(-1), pred_mlm.view(-1))
        l_mlm.extend(nptrue_mlm)
        p_mlm.extend(nppreds_mlm)


        loss.backward()

        optim.step()

        if schedular is not None:
            schedular.step()

    return losses_cls, losses_mlm, losses_pp, l_cls, p_cls, l_mlm, p_mlm, l_pp, p_pp

def validation(model_dibert, criterion_cls,criterion_mlm,criterion_pp, valid_iter, epoch):
    model_dibert.eval()
    losses_cls = []
    losses_mlm = []
    losses_pp = []

    l_cls = []
    p_cls = []
    l_pp = []
    p_pp = []
    l_mlm = []
    p_mlm = []
    softmax = nn.Softmax(dim=-1)
    print('\nValid_Epoch:', epoch)

    with torch.no_grad():
        for batch in tqdm.tqdm(valid_iter):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            token_type_ids = batch['token_type_ids'].cuda()
            #truelabel_pp = batch['indexes'].cuda()
            truelabel_pp = batch['parent_ids'].cuda()
            truelabel_mlm = batch['mask_ids'].cuda()
            truelabel_cls = batch['cls_label'].cuda()

            logits_cls, logits_mlm, logits_pp = model_dibert(input_ids, attention_mask, token_type_ids)
            ## if out dim is (bs x seqlen x numclass) -> (total_words_batch x numclass)
            ## if true label is (bs x seqlen) -> (total_words_batch)

            loss_mlm = criterion_mlm(logits_mlm.view(-1, model.Config.vocab_size), truelabel_mlm.view(-1))
            loss_cls = criterion_cls(logits_cls.view(-1, 2), truelabel_cls.view(-1))

            if (Config.is_dibert == True):
                #loss_pp = criterion_pp(logits_pp.view(-1, model.Config.max_len), truelabel_pp.view(-1))
                loss_pp = criterion_pp(logits_pp.view(-1, model.Config.vocab_size), truelabel_pp.view(-1))
                losses_pp.append(loss_pp.item())
                pred_pp = softmax(logits_pp).argmax(2)
                nptrue_pp, nppreds_pp = utils.prune_preds(truelabel_pp.view(-1), pred_pp.view(-1))
                l_pp.extend(nptrue_pp)
                p_pp.extend(nppreds_pp)

            losses_cls.append(loss_cls.item())
            losses_mlm.append(loss_mlm.item())

            # for now we are only interested in accuracy and f1 of the classification task
            l_cls.extend(truelabel_cls.cpu().detach().numpy())
            preds_cls = softmax(logits_cls).argmax(1)
            p_cls.extend(preds_cls.view(-1).cpu().detach().numpy())

            pred_mlm = softmax(logits_mlm).argmax(2)
            nptrue_mlm, nppreds_mlm = utils.prune_preds(truelabel_mlm.view(-1), pred_mlm.view(-1))
            l_mlm.extend(nptrue_mlm)
            p_mlm.extend(nppreds_mlm)

    return losses_cls, losses_mlm, losses_pp, l_cls, p_cls, l_mlm, p_mlm, l_pp, p_pp

def train_val(train_data, valid_data, model_path:str, trial = None, best_params = None):

    epochs = Config.n_epochs
    if (best_params is not None):
        print('selecting best_params')
        lrmain = best_params['lr']
    elif(trial is not None):
        print('selecting for trial')
        #lrmain = trial.suggest_loguniform('lrmain', 1e-6, 1e-4)
        #lrclassifier = trial.suggest_loguniform('lrclassifier', 1e-4, 1e-1)
        lrmain = trial.suggest_categorical('lr',[ 5e-5, 4e-5, 3e-5, 2e-5])
    else:
        lr = Config.lr

    train_iter = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
    valid_iter = DataLoader(valid_data, batch_size=Config.batch_size, shuffle=True)

    bert_tiny = model.dibert(model.Config)


    optimizer = AdamW(lr = lr, params=bert_tiny.parameters())

    #optimizer = torch.optim.Adam(lr = lr, params=bert_tiny.parameters(), weight_decay=0.01)
    training_steps = len(train_iter) * epochs
    warmUpSteps = (training_steps * Config.warmup)
    schedular = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmUpSteps, num_training_steps=training_steps)
    criterion_pp = nn.CrossEntropyLoss(ignore_index= 0)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_mlm = nn.CrossEntropyLoss(ignore_index= 0)
    bert_tiny.cuda()
    score = utils.score_cal()

    for epoch in range(epochs):
        losses_cls, losses_mlm, losses_pp, l_cls, p_cls, l_mlm, p_mlm, l_pp, p_pp = training(bert_tiny, optimizer, schedular, criterion_cls,criterion_mlm,criterion_pp, train_iter, epoch)
        print("------------------TRAIN---------------------------")
        f1,acc,loss = get_f1_acc_loss(l_cls,p_cls,losses_cls)
        print('train_weighted_f1_cls', f1)
        print('train_acc_cls', acc)
        print('train_loss_cls',sum(losses_cls)/len(losses_cls))
        print('\n')
        f1, acc, loss = get_f1_acc_loss(l_mlm, p_mlm, losses_mlm)
        print('train_weighted_f1_mlm', f1)
        print('train_acc_mlm', acc)
        print('train_loss_mlm', sum(losses_mlm) / len(losses_mlm))
        print('\n')
        if(l_pp):
            f1, acc, loss = get_f1_acc_loss(l_pp, p_pp, losses_pp)
            print('train_weighted_f1_pp', f1)
            print('train_acc_pp', acc)
            print('train_loss_pp', sum(losses_pp) / len(losses_pp))

        print("------------------VALIDATION---------------------------")
        losses_cls, losses_mlm, losses_pp, l_cls, p_cls, l_mlm, p_mlm, l_pp, p_pp = validation(bert_tiny, criterion_cls,criterion_mlm,criterion_pp, valid_iter, epoch)
        f1_cls, acc, loss = get_f1_acc_loss(l_cls, p_cls, losses_cls)
        print('valid_weighted_f1_cls', f1_cls)
        print('valid_acc_cls', acc)
        print('valid_loss_cls', sum(losses_cls) / len(losses_cls))
        print('\n')
        f1_mlm, acc, loss = get_f1_acc_loss(l_mlm, p_mlm, losses_mlm)
        print('valid_weighted_f1_mlm', f1_mlm)
        print('valid_acc_mlm', acc)
        print('valid_loss_mlm', sum(losses_mlm) / len(losses_mlm))
        print('\n')
        if (l_pp):
            f1_pp, acc, loss = get_f1_acc_loss(l_pp, p_pp, losses_pp)
            print('valid_weighted_f1_pp', f1_pp)
            print('valid_acc_pp', acc)
            print('valid_loss_pp', sum(losses_pp) / len(losses_pp))

        if (trial is None and best_params is None):
            print('-saving model-')
            state = {
                'epoch': epoch,
                'state_dict': bert_tiny.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            savepath = 'results/model/dibert_mlm_cls_pprediction_103_full_text'+str(epoch)+'.tar'
            torch.save(bert_tiny, savepath)



        #utils.classificationreport(valid_label, valid_preds)

    utils.save_json(model.model_config_to_dict(model.Config), 'results/model/dibert_mlm_cls_pprediction_model_103_full_text.json')
    utils.save_json(train_config_to_dict(Config),'results/model/dibert_mlm_cls_pprediction_train_103_full_text.json')
    return f1_cls+f1_mlm, score # tuning according to the last best validation accuracy
    #return sum(score.valid_acc)/len(score.valid_acc), score


def get_f1_acc_loss(true, preds, loss):
    f1, acc = utils.f1score(true, preds, 'weighted')
    return f1, acc, sum(loss)/len(loss)

def objective(train_data, valid_data, model_path, trial):
   try:
       acc, score = train_val(train_data=train_data, valid_data=valid_data, model_path=model_path, trial=trial, best_params=None)
   except:
       return 0
   return acc

def start_tuning(train_data, valid_data, model_path:str ,param_path:str, sampler='TPE'):
    if(sampler == 'TPE'):
        print('selecting tpe sampler')
        study = optuna.create_study(direction="maximize", sampler=TPESampler())
        study.optimize(lambda trial: objective(train_data, valid_data,model_path, trial), n_trials=30)

    elif(sampler == 'Grid'):
        print('selecting grid search sampler')
        search_space = {"lrmain": [5e-5, 4e-5, 3e-5, 2e-5], "drop_out": [0.0, 0.1, 0.2, 0.3]}
        study = optuna.create_study(direction="maximize", sampler=GridSampler(search_space))
        study.optimize(lambda trial:objective(train_data, valid_data, model_path, trial), n_trials=4 * 1 )
    elif(sampler == 'Grid_with_two_lr'):
        print('selecting grid search sampler 2lr')
        search_space = {"lrmain": [5e-5, 4e-5, 3e-5, 2e-5],'lrclassifier': [1e-3,1e-2, 1e-1], "drop_out": [0.0, 0.1, 0.2, 0.3]}
        study = optuna.create_study(direction="maximize", sampler=GridSampler(search_space))
        study.optimize(lambda trial: objective(train_data, valid_data, model_path, trial), n_trials=4 * 3 * 4)

    best_params = study.best_params
    utils.save_json(best_params, param_path)
    return best_params, study.best_trial

if __name__ == '__main__':

    param_path = 'results/params/model_2lr_sch.json'
    model_path = 'results/model/model_2lr_sch.tar'
    train_store_path = 'data/preprocessed/wiki-train103-fulltext.json'
    valid_store_path = 'data/preprocessed/wiki-valid103-fulltext.json'
    test_store_path = 'data/preprocessed/wiki-test103-fulltext.json'

    is_tune = False
    utils.set_seeds(Config.seed)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    train_data = dataset.Wiki2(train_store_path)
    valid_data = dataset.Wiki2(valid_store_path)
    print(len(train_data))
    print(len(valid_data))

    if(is_tune == False):
        _, score = train_val(train_data, valid_data, model_path, None, None)
        #utils.print_result(score, Config.epochs)
    elif(is_tune == True):
        #train_val(train_data, valid_data, model_path=model_path)
        best_params, best_trial = start_tuning(train_data, valid_data, model_path, param_path,'Grid_with_two_lr')






