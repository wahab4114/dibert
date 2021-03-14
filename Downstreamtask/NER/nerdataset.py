import torch
from transformers import BertTokenizer
from  Downstreamtask.NER.configner import NERConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class Data_preprocessing():
    def __init__(self, data_path):
        self.data_path = data_path

    def get_data_splits(self):
        df = pd.read_csv(self.data_path, encoding="latin-1")
        df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")
        enc_pos = preprocessing.LabelEncoder()
        enc_tag = preprocessing.LabelEncoder()
        df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
        df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])
        sentences = df.groupby("Sentence #")["Word"].apply(list).values
        pos = df.groupby("Sentence #")["POS"].apply(list).values
        tag = df.groupby("Sentence #")["Tag"].apply(list).values

        train_sen, test_sen, train_pos, test_pos, train_tag, test_tag = train_test_split(sentences, pos, tag, test_size=0.2, random_state=NERConfig.seed_1)
        valid_sen, test_sen, valid_pos, test_pos, valid_tag, test_tag = train_test_split(test_sen, test_pos, test_tag, test_size=0.5, random_state=NERConfig.seed_1)

        train_dataset = EntityDataset(train_sen, train_pos, train_tag)
        valid_dataset = EntityDataset(valid_sen, valid_pos, valid_tag)
        test_dataset = EntityDataset(test_sen, test_pos, test_tag)

        return train_dataset, valid_dataset, test_dataset, enc_pos, enc_tag

class EntityDataset:
    def __init__(self, texts, pos, tags):
        self.texts = texts
        self.pos = pos
        self.tags = tags
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tags = self.tags[item]
        pos = np.add(pos,1).tolist()

        ids = []
        target_pos = []
        target_tag = []

        for i, s in enumerate(text):
            inputs = self.tokenizer.encode(
                s,
                add_special_tokens=False
            )

            input_len = len(inputs)
            ids.extend(inputs)
            target_pos.extend([pos[i]] * input_len)
            target_tag.extend([tags[i]] * input_len)

        ids = ids[:NERConfig.seq_len - 2]
        target_pos = target_pos[:NERConfig.seq_len - 2]
        target_tag = target_tag[:NERConfig.seq_len - 2]

        ids = [101] + ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = NERConfig.seq_len - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_pos = target_pos + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }

if __name__ == "__main__":
    dp = Data_preprocessing('data/ner_dataset.csv')
    train, valid, test, enc_pos, enc_tag = dp.get_data_splits()
    print(train[0])
    print(enc_pos.classes_)

    #print(train[0])

