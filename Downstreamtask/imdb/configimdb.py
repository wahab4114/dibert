import os.path

class imdbConfig:
    tokenizer_name = 'bert-base-uncased'
    pretrained_model_name = 'bert-base-uncased'
    hidden_model_out = 768
    vocab_size = 30522
    #seq_len = 128 #from hugging face
    seq_len = 256
    batch_size = 128
    #batch_size = 32 #from hugging face
    epochs = 15
    seed_1 = 4114
    seed_2 = 391275
    seed_3 = 4664
    seed_4 = 117
    seed_5 = 360
    lr_bert = 1e-5
    lr_classifier = 0.001
    drop_out = 0.5
