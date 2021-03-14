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
            label.extend(truelabel_cls.cpu().detach().numpy())
            preds_cls = softmax(logits_cls).argmax(1)
            preds.extend(preds_cls.view(-1).cpu().detach().numpy())

    return label, preds


def main():

    model_path = 'results/full_text/dibert_LIAR_mlm_cls_pprediction_103_10_seed_4_epoch_7.tar'
    test_file = 'data/test.tsv'
    test_data = LiarDataset(test_file)
    print(len(test_data))
    Liar_Bert = torch.load(model_path)
    test_iter = DataLoader(test_data, batch_size=liarConfig.batch_size, shuffle=False)
    label, preds = testing(test_iter, Liar_Bert)
    f1, acc = f1score(label, preds, average='weighted')
    print(model_path)
    print('test_accuracy:', acc)
    print('test_weighted_f1:', f1)





if __name__ == "__main__":
    main()