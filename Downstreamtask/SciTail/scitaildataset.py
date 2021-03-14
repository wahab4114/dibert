import csv
from torch.utils.data import Dataset
from transformers import BertTokenizer
from Downstreamtask.SciTail.scitailconfig import scitailConfig
import torch


class SciTailData(Dataset):
    def __init__(self, file_name):
        self.filename = file_name
        self.data = self.read_tsv(self.filename)
        self.tokenizer = BertTokenizer.from_pretrained(scitailConfig.tokenizer_name)

    def __getitem__(self, item):
        t1, t2, label = self.data[item]
        encoded = self.tokenizer([t1],[t2],max_length=scitailConfig.seq_len, padding="max_length", truncation=True, return_tensors='pt')
        if(label=='neutral'):
            label = 0
        else:
            label = 1

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
                t1, t2, label = line.strip('\n').split('\t')
                data.append((t1.lower(), t2.lower(), label))
            return data



if __name__ == '__main__':
    train_file = 'data/scitail_1.0_train.tsv'
    test_file = 'data/scitail_1.0_test.tsv'
    valid_file = 'data/scitail_1.0_dev.tsv'
    data = SciTailData(train_file)
    print(len(data))
    print(data[0])

