from transformers import BertTokenizer, BertModel
import torch
import tqdm
import torch.nn as nn
from utils import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import *
import csv
from dataset import *
import pandas as pd
from Downstreamtask.LIAR.liardataset import LiarDataset
from Downstreamtask.LIAR.liarconfig import liarConfig




def testing(test_iter, model):
    model.eval()
    preds = []
    label = []
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        for batch in tqdm.tqdm(test_iter):
            input_ids = batch['input_ids'].cuda()
            attn_mask = batch['attention_mask'].cuda()
            truelabel_cls = batch['cls_label'].cuda()
            token_type_ids = batch['token_type_ids'].cuda()
            logits_cls = model(input_ids, attn_mask, token_type_ids)
            label.extend(truelabel_cls.view(-1).cpu().detach().numpy())
            preds_cls = softmax(logits_cls).argmax(1)
            preds.extend(preds_cls.view(-1).cpu().detach().numpy())

    return label, preds


def main():

    # model_path = 'results/full_text/seed_0/dibert_LIAR_mlm_cls_103_10_seed_0_epoch_5.tar'
    # test_file = 'data/test.tsv'
    # test_data = LiarDataset(test_file)
    # print(len(test_data))
    # Liar_Bert = torch.load(model_path)
    # test_iter = DataLoader(test_data, batch_size=liarConfig.batch_size, shuffle=False)
    # label, preds = testing(test_iter, Liar_Bert)
    # f1, acc = f1score(label, preds, average='weighted')
    # print(model_path)
    # print('test_accuracy:', acc)
    # print('test_weighted_f1:', f1)

    #this section is for qualitative study
    bert_model_path = 'results/full_text/seed_0/dibert_LIAR_mlm_cls_103_10_seed_0_epoch_5.tar'
    dibert_model_path = 'results/full_text/seed_0/dibert_LIAR_mlm_cls_pprediction_103_10_seed_0_epoch_10.tar'
    test_data = LiarDataset('data/test.tsv')
    print(len(test_data))
    Liar_Bert = torch.load(bert_model_path)
    test_iter = DataLoader(test_data, batch_size=liarConfig.batch_size, shuffle=False)

    label, bertpreds = testing(test_iter, Liar_Bert)
    Liar_Dibert = torch.load(dibert_model_path)
    label, dibertpreds = testing(test_iter, Liar_Dibert)

    print("label:", label)
    print("bert_preds:", bertpreds)
    print("dibert_preds:", dibertpreds)

    for i in range(len(label)):
        if (bertpreds[i] != dibertpreds[i]):
            print("index:",i)
            print("bertlabel:",bertpreds[i])
            print("dibertlabel:",dibertpreds[i])
            print("actual label:",label[i])


if __name__ == "__main__":
    main()

#index: 24
#bertlabel: 2
#dibertlabel: 1
#actual label: 1
#Says Charlie Crist is embroiled in a fraud case for steering taxpayer money to a de facto Ponzi scheme. -> barely-true, false, false

#index: 23
#bertlabel: 3
#dibertlabel: 5
#actual label: 4
#Marijuana is less toxic than alcohol. 'half-true', 'true', 'mostly-true'

#index: 25
#bertlabel: 1
#dibertlabel: 4
#actual label: 5
#Now, there was a time when someone like Scalia and Ginsburg got 95-plus votes. 'false', 'mostly-true', 'true'

# index: 50
# bertlabel: 2
# dibertlabel: 3
# actual label: 3
#Says Charlie Crist voted against raising the minimum wage. 'barely-true','half-true','half-true'

# index: 73
# bertlabel: 3
# dibertlabel: 5
# actual label: 4
#Every child born today inherits a $30,000 share in a national debt that stands at more than $13 trillion. 'half-true','true' 'mostly-true',

# index: 18
# bertlabel: 5
# dibertlabel: 4
# actual label: 5
#"Each year, 18,000 people die in America because they don't have health care.", 'True', 'mostly-true','True'

# index: 218
# bertlabel: 3
# dibertlabel: 4
# actual label: 3
#Two million more Latinos are in poverty today than when President Obama took his oath of office less than eight years ago. half-true, mostly-true, half-true