import random
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
from random import  shuffle
from random import random as rand
import spacy
import numpy as np
import math
import tqdm
import utils
from preprocessing import apply_preprocessing
import re

class PrepareDataWiki():
    def __init__(self,path, store_path):
        self.parser = spacy.load("en_core_web_lg")
        self.path = path
        self.store_path = store_path
        self.paragraphs = self._read_wiki(path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.vocab
        self.max_len = 512
        self.nsp_data = []
        self.mlm_data = []
        for paragraph in tqdm.tqdm(self.paragraphs):
            data1 = self._get_nsp_data_from_paragraph(paragraph, self.paragraphs, self.vocab, self.max_len)
            self.nsp_data.extend(data1)


        self.mlm_data = self._get_mlm_data()

    def get_parsed_parents(self, sentence):
        doc = self.parser(sentence)
        parents = []
        indexes = []
        text = []
        for token in doc:
            parents.append(token.head.text)
            indexes.append(token.head.i)
            text.append(token.text)

        return parents, indexes
    # passed one sentence, its parents and orignal indexes
    def get_parsed(self, parents, tokens, indexes):
        tokenized_tokens = []
        tokenized_parents = []
        tokenized_token_ids = []
        tokenized_parent_ids = []
        cumsum_arr = []
        new_idx = []
        for i,t in enumerate(tokens):
            l_tokens = self.tokenizer.tokenize(t)
            l_parents = self.tokenizer.tokenize(parents[i])
            idx = math.floor(len(l_parents)/2)
            parent = l_parents[idx]
            len_token = len(l_tokens)
            cumsum_arr.append(len_token)
            tokenized_parent_ids.extend(len_token * [self.tokenizer.vocab[parent]])
            tokenized_token_ids.extend([self.tokenizer.vocab[token] for token in l_tokens])
            tokenized_parents.extend([parent])
            tokenized_tokens.extend(l_tokens)
        cumsum = np.cumsum(cumsum_arr)
        for i in range(len(indexes)):
            if(indexes[i]==0):
                new_idx.extend(cumsum_arr[i] * [0])
            else:
                new_idx.extend(cumsum_arr[i]*[cumsum[indexes[i]-1]])
        return tokenized_tokens, tokenized_token_ids, tokenized_parents, tokenized_parent_ids, new_idx

    def _read_wiki(self, file_name):
        with open(file_name, 'r') as f:
            #lines = f.readlines()[:100000]
            lines = f.readlines()
        print(len(lines))

        # Uppercase letters are converted to lowercase ones
        paragraphs = [line.strip().lower().split(' . ')
                      for line in lines if len(line.split(' . ')) >= 2]
        return paragraphs

    def tokenize_text(self, text):
        doc = self.parser(self.preprocess(apply_preprocessing(text)))
        ## doing this becuase spacy tokenizes in a diff way
        txt = [token.text for token in doc]
        return ' '.join(txt)

    def preprocess(self, text):
        regex = re.compile('(\"\s)(.*?)(\s\")')  # remove whitespace around quotes
        text = re.sub(regex, '\"\\2\"', text)
        regex = re.compile('(\(\s)(.*?)(\s\))')  # remove whitespace inside brackets
        text = re.sub(regex, '\(\\2\)', text)
        regex = re.compile("(\s*)([\.\,\'\!\?\)])")  # remove whitespace before .,'!?)
        text = re.sub(regex, '\\2', text)
        regex = re.compile("\(\s*")  # remove whitespace after (
        text = re.sub(regex, '(', text)
        text = text.replace('@-@', '')  # remove @-@
        regex = re.compile("\s+")  # remove multiple whitespace
        text = re.sub(regex, ' ', text)
        return text


    def get_random_word_id(self, vocab):
        i = random.randint(0, len(vocab) - 1)
        return i

    def _get_nsp_data_from_paragraph(self, paragraph, paragraphs, vocab, max_len):
        nsp_data_from_paragraph = []
        for i in range(len(paragraph) - 1):
            tokens_a, tokens_b, is_next = self._get_next_sentence(
                paragraph[i], paragraph[i + 1], paragraphs)

            tokens_b = self.tokenize_text(tokens_b+' .')
            tokens_a = self.tokenize_text(tokens_a+' .')

            output = self.tokenizer([tokens_a], [tokens_b], max_length=self.max_len, padding='do_not_pad', truncation=True)
            nsp_data_from_paragraph.append((output, is_next, tokens_a+' '+tokens_b))
        return nsp_data_from_paragraph

    def _get_next_sentence(self, sentence, next_sentence, paragraphs):
        if random.random() < 0.5:
            is_next = True
        else:
            # `paragraphs` is a list of lists of lists
            next_sentence = random.choice(random.choice(paragraphs))
            is_next = False
        return sentence, next_sentence, is_next

    def _get_mlm_data(self):
        mlm_data = []
        for i in tqdm.tqdm(range(len(self.nsp_data))):
            output_dict, is_next, joined_text = self.nsp_data[i]
            joined_text = ' '.join(joined_text.split())
            parents, indexes = self.get_parsed_parents(joined_text)
            tokens = joined_text.split()
            tokens, token_ids, parents, parent_ids, indexes = self.get_parsed(parents, tokens, indexes)
            input_ids = output_dict['input_ids'][0]
            cand_pos = [i for i, token_id in enumerate(input_ids) if token_id != 101 and token_id != 102]
            n_pred = max(1, int(round(len(input_ids) * 0.15)))
            shuffle(cand_pos)
            cand_pos = cand_pos[:n_pred]
            mlm_ids = []
            mlm_tokens = []
            sep_middle_index = input_ids.index(102)
            indexes = np.add(indexes, 1).tolist() # because CLS and SEP is added to the input
            indexes.insert(0,0)
            indexes.insert(sep_middle_index,0)
            indexes.append(0)

            for i, idx in  enumerate(indexes):
                if(idx>=sep_middle_index):
                    indexes[i] = indexes[i] + 1

            for i, t in enumerate(input_ids):
                if (t == 101):
                    parent_ids.insert(i, 0)
                if (t == 102):
                    parent_ids.insert(i, 0)

            for i, t in enumerate(input_ids):
                if (i in cand_pos):
                    if rand() < 0.8:  # 80%
                        mlm_ids.append(input_ids[i])
                        #indexes[i] = 0
                        input_ids[i] = self.vocab['[MASK]']
                        parent_ids[i] = 0
                    elif rand() < 0.5:  # 10%
                        mlm_ids.append(input_ids[i])
                        #indexes[i] = 0
                        parent_ids[i] = 0
                        input_ids[i] = self.get_random_word_id(self.vocab)
                    else:
                        mlm_ids.append(input_ids[i])
                else:
                    mlm_ids.append(0)



            #print(len(input_ids), len(indexes), len(parent_ids))
            # mlm_data.append({'input_ids':input_ids, 'token_type_ids':output_dict['token_type_ids'][0],'attention_mask':output_dict['attention_mask'][0],'cls_label':is_next, 'mask_ids':mlm_ids, 'indexes': indexes,
            #                  'decoded':self.decode_ids(input_ids)})
            mlm_data.append({'input_ids': input_ids, 'token_type_ids': output_dict['token_type_ids'][0],
                             'attention_mask': output_dict['attention_mask'][0], 'cls_label': is_next,
                             'mask_ids': mlm_ids, 'parent_ids': parent_ids,
                             'decoded': self.decode_ids(input_ids),
                            'decoded_parents':self.decode_ids(parent_ids),'indexes': indexes})
        utils.save_json({'sentences':mlm_data}, self.store_path)
        return mlm_data

    def decode_ids(self, ids):
        return [self.tokenizer.decode([i]) for i in ids]



def main():
    train_data_path = 'data/wikitext-103-raw/wiki.train.raw'
    test_data_path = 'data/wikitext-103-raw/wiki.test.raw'
    valid_data_path = 'data/wikitext-103-raw/wiki.valid.raw'

    train_store_path = 'data/preprocessed/wiki-train103-fulltext.json'
    valid_store_path = 'data/preprocessed/wiki-valid103-fulltext.json'
    test_store_path = 'data/preprocessed/wiki-test103-fulltext.json'

    PrepareDataWiki(test_data_path, test_store_path)
    PrepareDataWiki(valid_data_path, valid_store_path)
    PrepareDataWiki(train_data_path, train_store_path)



if __name__ == '__main__':
    main()
