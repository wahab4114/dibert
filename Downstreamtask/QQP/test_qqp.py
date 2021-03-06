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
from Downstreamtask.QQP.qqpdataset import QQPdataset
from Downstreamtask.QQP.qqpconfig import qqpConfig




def testing(test_iter, model):
    model.eval()
    preds = []
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        for batch in tqdm.tqdm(test_iter):
            input_ids = batch['input_ids'].cuda()
            attn_mask = batch['attention_mask'].cuda()
            token_type_ids = batch['token_type_ids'].cuda()
            logits_cls = model(input_ids, attn_mask, token_type_ids)
            preds_cls = softmax(logits_cls).argmax(1)
            preds.extend(preds_cls.view(-1).cpu().detach().numpy())

    return  preds




def main():

    model_path = 'results/full_text/dibert_QQP_mlm_cls_pprediction_103_10_seed_4_epoch_7.tar'
    tsv_file = "results/full_text/dibert_QQP_mlm_cls_pprediction_103_10_seed_4_epoch_7.tsv"
    test_data = QQPdataset('test')
    print(len(test_data))
    qqp_bert = torch.load(model_path)
    test_iter = DataLoader(test_data, batch_size=qqpConfig.batch_size, shuffle=False)
    preds = testing(test_iter, qqp_bert)
    print(len(preds))
    idx = [i for i in range(len(preds))]
    result = {'index': idx, 'prediction': preds}
    pd.DataFrame(result).to_csv(tsv_file, index=False, sep='\t')


if __name__ == "__main__":
    main()