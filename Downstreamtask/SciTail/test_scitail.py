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
from Downstreamtask.SciTail.scidataset import SciTailData
from Downstreamtask.SciTail.scitailconfig import scitailConfig




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

    model_path = 'results/full_text/dibert_SCI_mlm_cls_pprediction_103_10_seed_0_epoch_5.tar'
    test_data = SciTailData('test')
    print(len(test_data))
    SciTail_bert = torch.load(model_path)
    test_iter = DataLoader(test_data, batch_size=scitailConfig.batch_size, shuffle=False)
    label, preds = testing(test_iter, SciTail_bert)
    f1, acc = f1score(label, preds, average='weighted')
    print(model_path)
    print('test_accuracy:', acc)
    print('test_weighted_f1:', f1)

    # this section is for qualitative study
    # bert_model_path = 'results/full_text/seed_3/dibert_SCI_mlm_cls_103_10_seed_3_epoch_11.tar'
    # dibert_model_path = 'results/full_text/seed_3/dibert_SCI_mlm_cls_pprediction_103_10_seed_3_epoch_15.tar'
    # test_data = SciTailData('test')
    # print(len(test_data))
    # scitail_Bert = torch.load(bert_model_path)
    # test_iter = DataLoader(test_data, batch_size=scitailConfig.batch_size, shuffle=False)
    #
    # label, bertpreds = testing(test_iter, scitail_Bert)
    # scitail_Bert = torch.load(dibert_model_path)
    # label, dibertpreds = testing(test_iter, scitail_Bert)
    #
    # print("label:", label)
    # print("bert_preds:", bertpreds)
    # print("dibert_preds:", dibertpreds)
    #
    # for i in range(len(label)):
    #     if (bertpreds[i] != dibertpreds[i]):
    #         print("index:",i)
    #         print("bertlabel:",bertpreds[i])
    #         print("dibertlabel:",dibertpreds[i])
    #         print("actual label:",label[i])





if __name__ == "__main__":
    main()

#index: 1
#bertlabel: 1
#dibertlabel: 0
#actual label: 0
#Based on the list provided of the uses of substances 1-7, estimate the pH of each unknown and record the number in the data table in the estimated pH column.
# If a substance has a ph value greater than 7,that indicates that it is base. "entails","neutral","neutral"

# index: 48
# bertlabel: 0
# dibertlabel: 1
# actual label: [1]
#Front- The boundary between two different air masses.
# In weather terms, the boundary between two air masses is called front."neutral","entails","entails"

# index: 12
# bertlabel: 0
# dibertlabel: 1
# actual label: [1]
# Vertebrates are named for vertebrae, the series of bones that make up the vertebral column or backbone.	Backbone is another name for the vertebral column.
# "neutral","entails","entails"

# index: 15
# bertlabel: 0
# dibertlabel: 1
# actual label: [0]
#Neptune will be the farthest planet from the Sun until 1999.	The eighth planet from our sun is neptune.
# "neutral", "entrails", "neutral"