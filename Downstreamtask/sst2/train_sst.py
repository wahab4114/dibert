from optuna.samplers import TPESampler, GridSampler

from Downstreamtask.sst2.configsst import sstConfig
import numpy as np
import torch.nn as nn
import torch
import time
import torch.optim as optim
from utils import *
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from Downstreamtask.sst2.sstmodel import Bert_base
import tqdm
import optuna
from torch.utils.data import Dataset, DataLoader
from Downstreamtask.sst2.dataset import SstDataset

def training(model, optim, criterion_cls, train_iter, epoch):

    model.train()
    losses = []
    label = []
    preds = []
    softmax = nn.Softmax(dim = -1)
    print('\nTrain_Epoch:', epoch)
    for batch in tqdm.tqdm(train_iter):
        optim.zero_grad()
        input_ids = batch['input_ids'].cuda()
        attn_mask = batch['attention_mask'].cuda()
        truelabel_cls = batch['cls_label'].cuda()

        logits_cls = model(input_ids, attn_mask)
        ## if out dim is (bs x seqlen x numclass) -> (total_words_batch x numclass)
        ## if true label is (bs x seqlen) -> (total_words_batch)
        loss_cls = criterion_cls (logits_cls.view(-1, 2), truelabel_cls.view(-1, ))
        loss = loss_cls
        losses.append(loss.item())
        #for now we are only interested in accuracy and f1 of the classification task
        label.extend(truelabel_cls.cpu().detach().numpy())
        preds_cls = softmax(logits_cls).argmax(1)
        preds.extend(preds_cls.view(-1).cpu().detach().numpy())

        loss.backward()

        optim.step()

    return losses, label, preds

def validation(model, criterion_cls, valid_iter, epoch):
    model.eval()
    losses = []
    label = []
    preds = []
    softmax = nn.Softmax(dim=-1)
    print('\nValid_Epoch:', epoch)

    with torch.no_grad():
        for batch in tqdm.tqdm(valid_iter):
            input_ids = batch['input_ids'].cuda()
            attn_mask = batch['attention_mask'].cuda()
            truelabel_cls = batch['cls_label'].cuda()

            logits_cls = model(input_ids, attn_mask)

            loss_cls = criterion_cls(logits_cls.view(-1, 2), truelabel_cls.view(-1, ))
            loss = loss_cls
            losses.append(loss.item())
            # for now we are only interested in accuracy and f1 of the classification task
            label.extend(truelabel_cls.cpu().detach().numpy())
            preds_cls = softmax(logits_cls).argmax(1)
            preds.extend(preds_cls.view(-1).cpu().detach().numpy())

    return losses, label, preds

def train_val(train_data, valid_data, model_path:str,trial=None, best_params=None):

    epochs = sstConfig.epochs
    if (best_params is not None):
        print('selecting best_params')
        lrmain = best_params['lrmain']
        #lrclassifier = best_params['lrclassifier']
        drop_out = best_params['drop_out']
    elif(trial is not None):
        print('selecting for trial')
        lrmain = trial.suggest_categorical('lrmain',[ 5e-5, 4e-5, 3e-5, 2e-5])
        drop_out = trial.suggest_categorical('drop_out', [0.0, 0.1, 0.2, 0.3])
        #lrmain = trial.suggest_categorical('lrmain',[5e-5, 3e-5, 2e-5])
        #drop_out = trial.suggest_categorical('drop_out', [0.1])
    else:
        lrmain = sstConfig.lr_bert
        #lrclassifier = sstConfig.lr_classifier
        drop_out = sstConfig.drop_out

    train_iter = DataLoader(train_data, batch_size=sstConfig.batch_size, shuffle=True)
    valid_iter = DataLoader(valid_data, batch_size=sstConfig.batch_size, shuffle=True)
    path_1 = '../../results/model/wiki103_mlm_cls_full_epochs10/dibert_mlm_cls_103_full_text9.tar'
    path_2 = '../../results/model/wiki103_mlm_cls_pprediction_full_epochs10/dibert_mlm_cls_pprediction_103_full_text9.tar'

    bert_pretrained = torch.load(path_2)
    model = Bert_base(pretrained_model= bert_pretrained, hidden_out=sstConfig.hidden_model_out, drop_out= drop_out)

    bert_param = []
    classifier = []
    # in case of fine tuning we want diff learning rates for pretrained bert and diff for additional bert layer
    # for m in model.parameters():
    #     if 2 in m.size():  # classification layer additional params
    #         classifier.append(m)
    #     else:
    #         bert_param.append(m)
    # optimizer = AdamW([
    #       {'params': bert_param, 'lr': lrmain},
    #       {'params': classifier, 'lr': lrclassifier}])

    optimizer = AdamW(model.parameters(), lr=lrmain)

    criterion_cls = nn.CrossEntropyLoss()
    model.cuda()
    score = score_cal()

    for epoch in range(epochs):
        train_losses, label, preds = training(model, optimizer, criterion_cls, train_iter, epoch)
        f1, acc = f1score(label, preds, 'weighted')
        score.train_f1.append(f1)
        score.train_acc.append(acc)
        score.train_loss.append(sum(train_losses)/len(train_losses))
        print('train_weighted_f1', f1)
        print('train_acc', acc)

        valid_loss, valid_label, valid_preds = validation(model, criterion_cls, valid_iter, epoch)
        valid_f1, valid_acc = f1score(valid_label, valid_preds, 'weighted')
        score.valid_f1.append(valid_f1)
        score.valid_acc.append(valid_acc)
        score.valid_loss.append(sum(valid_loss) / len(valid_loss))

        print('valid_weighted_f1:', valid_f1)
        print('valid_acc:', valid_acc)

        classificationreport(valid_label, valid_preds)

        if(trial is None and best_params is not None):
            print('-saving model-')
            torch.save(model, 'results/full_text/dibert_mlm_cls_pprediction_full_text_103_10_seed_'+str(4)+'_epoch_'+str(epoch+1)+'.tar')

    return valid_acc, score # tuning according to the last best validation accuracy
    #return sum(score.valid_acc)/len(score.valid_acc), score

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
        #search_space = {"lrmain": [5e-5, 3e-5, 2e-5], "drop_out": [0.1]}
        search_space = {"lrmain": [5e-5, 4e-5, 3e-5, 2e-5], "drop_out": [0.0, 0.1, 0.2, 0.3]}
        print(search_space)
        study = optuna.create_study(direction="maximize", sampler=GridSampler(search_space))
        study.optimize(lambda trial:objective(train_data, valid_data, model_path, trial), n_trials=4*4 )
    elif(sampler == 'Grid_with_two_lr'):
        print('selecting grid search sampler 2lr')
        search_space = {"lrmain": [5e-5, 4e-5, 3e-5, 2e-5],'lrclassifier': [1e-3,1e-2, 1e-1], "drop_out": [0.0, 0.1, 0.2, 0.3]}
        study = optuna.create_study(direction="maximize", sampler=GridSampler(search_space))
        study.optimize(lambda trial: objective(train_data, valid_data, model_path, trial), n_trials=4 * 3 * 4)

    best_params = study.best_params
    save_json(best_params, param_path)
    return best_params, study.best_trial

if __name__ == '__main__':

    param_path = 'results/full_text/params/dibert_mlm_cls_pprediction_full_text.json'
    model_path = 'results/dibert_mlm_cls_58_seed_'+str(1)+'.tar'

    is_tune = False
    #np.random.seed(sstConfig.seed_4)
    #torch.manual_seed(sstConfig.seed_4)
    #random.seed(sstConfig.seed_4)
    #torch.manual_seed(sstConfig.seed_4)
    #torch.cuda.manual_seed_all(sstConfig.seed_4)

    device = torch.device("cuda:0")

    train_data = SstDataset('train')
    valid_data = SstDataset('validation')
    test_data = SstDataset('test')
    print(len(train_data))
    print(len(valid_data))
    print(len(test_data))

    if(is_tune == False):
        best_params = load_json(param_path)
        #best_params = {"lrmain": 4e-05, "drop_out": 0.0}
        print(best_params)
        _, score = train_val(train_data, valid_data, model_path, None, best_params)
        print_result(score, sstConfig.epochs)
    elif(is_tune == True):
        #train_val(train_data, valid_data, model_path=model_path)
        best_params, best_trial = start_tuning(train_data, valid_data, model_path, param_path, 'Grid')



