import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import NamedTuple

class Config():
    vocab_size: int = 30522  # Size of Vocabulary
    hidden_size: int = 768  # Dimension of Hidden Layer in Transformer Encoder
    num_hidden_layers: int = 2  # Numher of Hidden Layers
    num_attention_heads: int = 2  # Numher of Heads in Multi-Headed Attention Layers
    intermediate_size: int = 768 * 4  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    # activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    max_len: int = 512  # Maximum Length for Positional Embeddings
    n_segments: int = 2  # Number of Sentence Segments
    attention_probs_dropout_prob: int = 0.1

def model_config_to_dict(Config):
    return {'vocab_size': Config.vocab_size,
                'hidden_size': Config.hidden_size,
                'num_hidden_layers': Config.num_hidden_layers,
                'num_attention_heads': Config.num_attention_heads,
                'intermediate_size': Config.intermediate_size, 'max_len': Config.max_len,
                'n_segments': Config.n_segments, 'attention_probs_dropout_prob':Config.attention_probs_dropout_prob}


class dibert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = BertConfig(vocab_size=config.vocab_size,hidden_size=config.hidden_size,
                        num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads,
                        intermediate_size=config.intermediate_size, attention_probs_dropout_prob=config.attention_probs_dropout_prob
                        , max_position_embeddings=config.max_len,
                        type_vocab_size=config.n_segments)
        self.bert = BertModel(self.config)
        print(self.bert.config)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.lm = nn.Linear(config.hidden_size, config.vocab_size)
        #self.pp = nn.Linear(config.hidden_size, config.max_len)
        self.pp = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        h, h_pooled = self.bert(input_ids, attention_mask, token_type_ids)

        logits_cls = self.classifier(h_pooled)
        logits_lm =  self.lm(h)
        logits_pp = self.pp(h)
        #print(logits_cls.size(), logits_lm.size(), logits_pp.size())
        return logits_cls, logits_lm, logits_pp




def main():
    pass
if __name__ == '__main__':
    main()