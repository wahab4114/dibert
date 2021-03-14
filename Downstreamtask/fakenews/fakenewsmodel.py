import torch.nn as nn
from Downstreamtask.fakenews.configfakenews import fakenewsConfig
from model import dibert

class ClassificationHead(nn.Module):
    def __init__(self, hidden_out , drop_out):
        super().__init__()
        self.hidden = hidden_out
        self.drop_out = nn.Dropout(p=drop_out)
        self.cls_layer = nn.Linear(self.hidden, 2)
    def forward(self, x):
        x = self.drop_out(x)
        x = self.cls_layer(x)
        return x


class Bert_fakenews(nn.Module): #for checking sota results
    def __init__(self, pretrained_model, hidden_out = fakenewsConfig.hidden_model_out, seq_len = fakenewsConfig.seq_len, drop_out = fakenewsConfig.drop_out):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.hidden = hidden_out
        self.cls_layer = ClassificationHead(self.hidden, drop_out)

    def forward(self, input_ids, attention_mask):
        o1, o2 = self.pretrained_model.bert(input_ids, attention_mask)
        out_cls = self.cls_layer(o2)
        return out_cls