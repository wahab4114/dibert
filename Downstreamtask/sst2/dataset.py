from torch.utils.data import Dataset, DataLoader
from Downstreamtask.sst2.configsst import sstConfig
from transformers import BertTokenizer
import torch
import csv

class SstDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = self.read_csv()
        self.tokenizer = BertTokenizer.from_pretrained(sstConfig.tokenizer_name)


    def __getitem__(self, index):
        sentence, label =  self.data[index]
        output = self.tokenizer(sentence, max_length=sstConfig.seq_len, padding='max_length', truncation=True, return_tensors='pt')
        #print(output)
        return {'input_ids':output['input_ids'][0],
         'attention_mask':output['attention_mask'][0],
         'cls_label':torch.LongTensor([label])
         }
    def __len__(self):
        return len(self.data)

    def read_csv(self):
        print('--loading csv file--')
        sentences = []
        with open(self.path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            next(csvreader, None)  # skip header
            for line in csvreader:
                label, sentence = line
                sentences.append((sentence,int(label)))
        return sentences





def main():
    path = 'data/val.csv'
    data = SstDataset(path)
    print(data[0])
    print(len(data))
if __name__ == "__main__":
    main()

