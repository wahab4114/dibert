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
from Downstreamtask.QNLI.qnlidataset import QNLIdataset
from Downstreamtask.QNLI.qnliconfig import qnliConfig




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

    model_path = 'results/full_text/dibert_QNLI_mlm_cls_pprediction_103_10_seed_0_epoch_11.tar'
    tsv_file = "results/full_text/dibert_QNLI_mlm_cls_pprediction_103_10_seed_0_epoch_11.tsv"
    test_data = QNLIdataset('test')
    print(len(test_data))
    qnli_bert = torch.load(model_path)
    test_iter = DataLoader(test_data, batch_size=qnliConfig.batch_size, shuffle=False)
    preds = testing(test_iter, qnli_bert)
    print(len(preds))
    string_preds = ['entailment' if pred == 0 else 'not_entailment' for pred in preds]
    idx = [i for i in range(len(preds))]
    result = {'index': idx, 'prediction': string_preds}
    pd.DataFrame(result).to_csv(tsv_file, index=False, sep='\t')


if __name__ == "__main__":
    main()