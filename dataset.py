from torch.utils.data import Dataset, DataLoader
import train
import model
import utils
import torch


class Wiki2(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = utils.load_json(path)
        self.data = self.data['sentences']

    def add_padding(self, ids):
        if (model.Config.max_len > len(ids)):
            padding_len = model.Config.max_len - len(ids)
            ids = ids + padding_len * [0]
        if (model.Config.max_len < len(ids)):
            ids = ids[:model.Config.max_len]
        return torch.LongTensor(ids)

    def __getitem__(self, item):
        input_ids = self.add_padding(self.data[item]['input_ids'])
        token_type_ids = self.add_padding(self.data[item]['token_type_ids'])
        attention_mask = self.add_padding(self.data[item]['attention_mask'])
        cls_label = torch.LongTensor([int(self.data[item]['cls_label'])])
        mask_ids = self.add_padding(self.data[item]['mask_ids'])
        parent_ids = self.add_padding(self.data[item]['parent_ids'])
        #indexes = self.add_padding(self.data[item]['indexes'])

        return {'input_ids': input_ids, 'token_type_ids':token_type_ids,
                'attention_mask':attention_mask, 'cls_label':cls_label,
                'mask_ids':mask_ids, 'parent_ids':parent_ids}

    def __len__(self):
        return len(self.data)

def main():
    train_store_path = 'data/preprocessed/wiki-train103-5pp.json'
    valid_store_path = 'data/preprocessed/wiki-valid103-5pp.json'
    test_store_path = 'data/preprocessed/wiki-test103-5pp.json'

    wiki = Wiki2(train_store_path)
    print(len(wiki))
    print(wiki[0]['input_ids'])
    print(wiki[0]['token_type_ids'])
    print(wiki[0]['attention_mask'])
    print(wiki[0]['cls_label'])
    print(wiki[0]['mask_ids'])
    print(wiki[0]['parent_ids'])

if __name__ == '__main__':
    main()