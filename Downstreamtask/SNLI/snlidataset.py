from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer
from Downstreamtask.SciTail.scitailconfig import scitailConfig
import torch
import tqdm
import numpy as np

class SNLIData(Dataset):
    def __init__(self, split_name = 'validation'):
        self.splitname = split_name
        self.sen1, self.sen2, self.label = self.read_tsv(self.splitname)
        self.tokenizer = BertTokenizer.from_pretrained(scitailConfig.tokenizer_name)

    def __getitem__(self, item):
        t1, t2, label = self.sen1[item], self.sen2[item], self.label[item]
        encoded = self.tokenizer([t1],[t2],max_length=scitailConfig.seq_len, padding="max_length", truncation=True, return_tensors='pt')

        return {'input_ids': encoded['input_ids'][0], 'token_type_ids':encoded['token_type_ids'][0],
                    'attention_mask': encoded['attention_mask'][0],
                    'cls_label': torch.LongTensor([label])
                    }

    def __len__(self):
        return len(self.sen1)

    def read_tsv(self, split_name='validation'):

        dataset = load_dataset('snli', split=split_name)
        # removing -1 labels
        indices = [i for i, x in enumerate(dataset['label']) if x == -1]
        premise = np.delete(dataset['premise'], indices)
        hypothesis = np.delete(dataset['hypothesis'], indices)
        label = np.delete(dataset['label'], indices)
        return premise, hypothesis, label





if __name__ == '__main__':
    split_name = 'train'
    data = SNLIData(split_name)
    print(len(data))
    print(data[0])


    #print(dataset['test']['premise'])

