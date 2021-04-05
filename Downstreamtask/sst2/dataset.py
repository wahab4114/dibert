from torch.utils.data import Dataset, DataLoader
from Downstreamtask.sst2.configsst import sstConfig
from transformers import BertTokenizer
import torch
from datasets import load_dataset
import csv

class SstDataset(Dataset):
    def __init__(self, split_name):
        self.splitname = split_name
        self.sen1, self.label = self.read_tsv(self.splitname)
        self.tokenizer = BertTokenizer.from_pretrained(sstConfig.tokenizer_name)


    def __getitem__(self, index):
        output = self.tokenizer(self.sen1[index], max_length=sstConfig.seq_len, padding='max_length', truncation=True, return_tensors='pt')
        #print(output)
        return {'input_ids':output['input_ids'][0],
         'attention_mask':output['attention_mask'][0],
         'cls_label':torch.LongTensor([self.label[index]])
         }
    def __len__(self):
        return len(self.sen1)

    def read_tsv(self, split_name='validation'):
        dataset = load_dataset('glue', 'sst2', split=split_name)
        return dataset['sentence'], dataset['label']





def main():
    split_name = 'train'
    data = SstDataset(split_name)
    print(len(data))
    print(data[0])
if __name__ == "__main__":
    main()

