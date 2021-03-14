import torch.nn as nn
from Downstreamtask.NER.configner import NERConfig
from model import dibert

class NERHead(nn.Module):
    def __init__(self, hidden_out , drop_out):
        super().__init__()
        self.hidden = hidden_out
        self.drop_out = nn.Dropout(p=drop_out)
        self.cls_layer = nn.Linear(self.hidden, NERConfig.num_labels_pos)
    def forward(self, x):
        x = self.drop_out(x)
        x = self.cls_layer(x)
        return x


class Bert_ner(nn.Module): #for checking sota results
    def __init__(self, pretrained_model, hidden_out = NERConfig.hidden_model_out, seq_len = NERConfig.seq_len, drop_out = NERConfig.drop_out):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.hidden = hidden_out
        self.ner_layer = NERHead(self.hidden, drop_out)

    def forward(self, input_ids, attention_mask):
        o1, o2 = self.pretrained_model.bert(input_ids, attention_mask)
        #[bs x seqlen x hiddendim]
        out_cls = self.ner_layer(o1)
        return out_cls