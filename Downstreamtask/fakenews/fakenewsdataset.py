import csv
from torch.utils.data import Dataset
from transformers import BertTokenizer
from Downstreamtask.fakenews.configfakenews import fakenewsConfig
import torch
import preprocessor as p


class FakeNewsDataset(Dataset):
    def __init__(self, file_name):
        self.filename = file_name
        self.data = self.read_csv(self.filename)
        self.tokenizer = BertTokenizer.from_pretrained(fakenewsConfig.tokenizer_name)

    def __getitem__(self, item):
        text, label = self.data[item]
        encoded = self.tokenizer(text,max_length=fakenewsConfig.seq_len, padding="max_length", truncation=True, return_tensors='pt')
        if(label=='fake'):
            label = 1
        else:
            label = 0

        return {'input_ids': encoded['input_ids'][0], 'token_type_ids':encoded['token_type_ids'][0],
                    'attention_mask': encoded['attention_mask'][0],
                    'cls_label': torch.LongTensor([label])
                    }

    def __len__(self):
        return len(self.data)

    def read_csv(self, path):
        p.set_options(p.OPT.URL, p.OPT.EMOJI)
        sentences = []
        with open(path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            next(csvreader, None)  # skip header
            for line in csvreader:
                id, sentence, label = line
                sentence = p.clean(sentence)  # tweet preprocessor deleting urls and emojis
                sentences.append((sentence, label))
        return sentences



if __name__ == '__main__':
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'
    valid_file = 'data/valid.csv'
    data = FakeNewsDataset(train_file)
    print(len(data))
    print(data[0])


