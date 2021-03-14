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
from Downstreamtask.NER.nerdataset import Data_preprocessing
from Downstreamtask.NER.configner import NERConfig




def testing(test_iter, model):
    model.eval()
    preds = []
    label = []
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        for batch in tqdm.tqdm(test_iter):
            input_ids = batch['input_ids'].cuda()
            attn_mask = batch['attention_mask'].cuda()
            truelabel_pos = batch['target_pos'].cuda()
            logits_pos = model(input_ids, attn_mask)
            preds_pos = softmax(logits_pos).argmax(2)
            nptrue_pos, nppreds_pos = prune_preds(truelabel_pos.view(-1), preds_pos.view(-1))
            label.extend(nptrue_pos)
            preds.extend(nppreds_pos)

    return label, preds


def main():
    model_path = 'results/full_text/dibert_NER_mlm_cls_pprediction_103_10_seed_1_epoch_14.tar'

    dp = Data_preprocessing('data/ner_dataset.csv')
    train_data, valid_data, test_data, enc_pos, enc_tag = dp.get_data_splits()
    print(len(train_data))
    print(len(valid_data))
    test_iter = DataLoader(test_data, batch_size=NERConfig.batch_size, shuffle=False)
    ner_bert = torch.load(model_path)

    label, preds = testing(test_iter, ner_bert)
    f1, acc = f1score(label, preds, average='weighted')
    print(model_path)
    print('test_accuracy:', acc)
    print('test_weighted_f1:', f1)


if __name__ == "__main__":
    main()