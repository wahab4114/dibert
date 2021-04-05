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
from Downstreamtask.sst2.dataset import SstDataset
from Downstreamtask.sst2.configsst import sstConfig




def testing(test_iter, model):
    model.eval()
    preds = []

    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        for batch in tqdm.tqdm(test_iter):
            input_ids = batch['input_ids'].cuda()
            attn_mask = batch['attention_mask'].cuda()
            logits_cls = model(input_ids, attn_mask)
            preds_cls = softmax(logits_cls).argmax(1)
            preds.extend(preds_cls.view(-1).cpu().detach().numpy())

    return preds


def main():
    model_path = 'results/full_text/dibert_mlm_cls_pprediction_full_text_103_10_seed_0_epoch_13.tar'
    tsv_file = 'results/full_text/dibert_mlm_cls_pprediction_full_text_103_10_seed_0_epoch_13.tsv'
    test_data  = SstDataset('test')

    sst_bert = torch.load(model_path)

    test_iter = DataLoader(test_data, batch_size=sstConfig.batch_size, shuffle=False)

    preds = testing(test_iter, sst_bert)
    print(len(preds))
    idx = [i for i in range(len(preds))]
    result = {'index': idx, 'prediction': preds}
    pd.DataFrame(result).to_csv(tsv_file, index=False, sep='\t')


if __name__ == "__main__":
    main()