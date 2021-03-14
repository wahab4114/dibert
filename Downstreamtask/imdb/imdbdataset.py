from torch.utils.data import Dataset, DataLoader
from Downstreamtask.imdb.configimdb import imdbConfig
from transformers import BertTokenizer
import torch
import csv
import re
from sklearn.model_selection import train_test_split

class IMDBdataset(Dataset):

    @classmethod
    def get_data_splits(cls, path):
        data = cls.read_csv(path)
        train_data, test_data = train_test_split(data, test_size=0.5, random_state=imdbConfig.seed_1)
        valid_data, test_data = train_test_split(test_data, test_size=0.5, random_state=imdbConfig.seed_1)
        return train_data, valid_data, test_data

    def __init__(self, data):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained(imdbConfig.tokenizer_name)


    def __getitem__(self, index):
        sentence, label = self.data[index]
        output = self.tokenizer(sentence, max_length=imdbConfig.seq_len, padding='max_length', truncation=True,
                                    return_tensors='pt')
        return {'input_ids': output['input_ids'][0],
                    'attention_mask': output['attention_mask'][0],
                    'cls_label': torch.LongTensor([label])
                    }

    def __len__(self):
        return len(self.data)

    @classmethod
    def cleanhtml(cls,raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext.lower()

    @classmethod
    def read_csv(cls,path):
        print('--loading csv file--')
        sentences = []
        with open(path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            next(csvreader, None)  # skip header
            for line in csvreader:
                sentence, label = line
                sentence = cls.cleanhtml(sentence)
                sentences.append((sentence,int(label)))
        return sentences





def main():
    path = 'data/IMDB_datav2.csv'
    train_data, valid_data, test_data = IMDBdataset.get_data_splits(path)
    data = IMDBdataset(test_data)
    print(len(data))
    print(data[0])




if __name__ == "__main__":
    main()

