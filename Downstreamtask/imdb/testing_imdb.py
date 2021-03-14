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
from Downstreamtask.imdb.imdbdataset import IMDBdataset
from Downstreamtask.imdb.configimdb import imdbConfig




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
            logits_cls = model(input_ids, attn_mask)
            label.extend(truelabel_cls.cpu().detach().numpy())
            preds_cls = softmax(logits_cls).argmax(1)
            preds.extend(preds_cls.view(-1).cpu().detach().numpy())

    return label, preds


def main():
    model_path = 'results/full_text/dibert_mlm_cls_pprediction_full_text_103_10_seed_4_epoch_1.tar'

    path = 'data/IMDB_datav2.csv'
    train_data_split, valid_data_split, test_data_split = IMDBdataset.get_data_splits(path)
    test_data = IMDBdataset(test_data_split)
    print(len(test_data))
    imdb_bert = torch.load(model_path)
    test_iter = DataLoader(test_data, batch_size=imdbConfig.batch_size, shuffle=False)

    label, preds = testing(test_iter, imdb_bert)
    f1, acc = f1score(label, preds, average='weighted')
    print(model_path)
    print('test_accuracy:', acc)
    print('test_weighted_f1:', f1)


if __name__ == "__main__":
    main()