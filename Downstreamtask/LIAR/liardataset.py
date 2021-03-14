import csv
from torch.utils.data import Dataset
from transformers import BertTokenizer
from Downstreamtask.LIAR.liarconfig import liarConfig
import torch
import numpy as np

class LiarDataset(Dataset):
    def __init__(self, file_name):
        self.filename = file_name
        self.data = self.read_tsv(self.filename)
        self.tokenizer = BertTokenizer.from_pretrained(liarConfig.tokenizer_name)

    def __getitem__(self, item):
        label, statement,sub, speaker, speaker_job, state, party, venue = self.data[item]
        text= statement+' '+speaker+' '+speaker_job+' '+state+' '+party+' '+venue
        encoded = self.tokenizer(text, max_length=liarConfig.seq_len, padding="max_length", truncation=True, return_tensors='pt')
        if(label=='pants-fire'):
            label = 0
        elif(label=='false'):
            label = 1
        elif(label == 'barely-true'):
            label = 2
        elif(label == 'half-true'):
            label = 3
        elif (label == 'mostly-true'):
            label = 4
        elif (label == 'true'):
            label = 5

        return {'input_ids': encoded['input_ids'][0], 'token_type_ids':encoded['token_type_ids'][0],
                    'attention_mask': encoded['attention_mask'][0],
                    'cls_label': torch.LongTensor([label])
                    }

    def __len__(self):
        return len(self.data)

    def read_tsv(self, filename):
        with open(filename, "r") as fd:
            data = []
            for line in fd:
                id, label, statement,sub, speaker, speaker_job,state, party, _,_,_,_,_, venue = line.strip('\n').split('\t')
                data.append((label.lower(), statement.lower(), sub.lower(), speaker.lower(), speaker_job.lower(), state.lower(), party.lower(), venue.lower()))
            return data

def read_tsv(filename):
        with open(filename, "r") as fd:
            data = []
            labels = []
            for line in fd:
                id, label, statement,sub, speaker, speaker_job,state, party, _,_,_,_,_, venue = line.strip('\n').split('\t')
                data.append((label.lower(), statement.lower(), sub.lower(), speaker.lower(), speaker_job.lower(), state.lower(), party.lower(), venue.lower()))
                labels.append(label.lower())
            print(np.unique(labels))
            return data

if __name__ == '__main__':
    train_file = 'data/train.tsv'
    test_file = 'data/test.tsv'
    valid_file = 'data/valid.tsv'

    dataset = LiarDataset(valid_file)
    print(dataset[20])