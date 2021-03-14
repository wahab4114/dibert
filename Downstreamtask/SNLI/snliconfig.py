import os.path

class snliConfig:
    tokenizer_name = 'bert-base-uncased'
    pretrained_model_name = 'bert-base-uncased'
    hidden_model_out = 768
    vocab_size = 30522
    #seq_len = 128 #from hugging face
    seq_len = 128
    batch_size = 128
    #batch_size = 32 #from hugging face
    epochs = 15
    lr_bert = 1e-5
    lr_classifier = 0.001
    drop_out = 0.5