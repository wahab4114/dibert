
from sklearn.metrics import *
from matplotlib import pyplot
import pickle
import os
import random
import logging
import json
import numpy as np
import torch

def save_json(params, path):
    with open(path, 'w') as fp:
        json.dump(params, fp)

def load_json(path):
    with open(path, 'r') as fp:
        params = json.load(fp)
    return params


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device






def get_random_word(vocab_words):
    i = random.randint(0, len(vocab_words)-1)
    return vocab_words[i]

def get_logger(name, log_path):
    "get logger"
    logger = logging.getLogger(name)
    fomatter = logging.Formatter(
        '[ %(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    if not os.path.isfile(log_path):
        f = open(log_path, "w+")

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)

    #streamHandler = logging.StreamHandler()
    #streamHandler.setFormatter(fomatter)
    #logger.addHandler(streamHandler)

    logger.setLevel(logging.DEBUG)
    return logger


def f1score(y_true, y_pred, average = 'macro'):
    f1 = f1_score(y_true, y_pred, average=average)
    acc = accuracy_score(y_true, y_pred)
    return f1, acc

def classificationreport(y_true, y_pred):
    print(classification_report(y_true, y_pred))

class score_cal:
    def __init__(self):
        self.train_f1 = []
        self.valid_f1 = []
        self.train_acc = []
        self.valid_acc = []
        self.train_loss = []
        self.valid_loss = []

def print_result(score:score_cal, epochs):
    epoch_vals = [i + 1 for i in range(epochs)]
    pyplot.subplot(311)
    pyplot.title("Loss")
    pyplot.plot(epoch_vals, score.train_loss, label='train')
    pyplot.plot(epoch_vals, score.valid_loss, label='valid')
    pyplot.legend()
    pyplot.xticks(epoch_vals)

    if(score.train_f1 != []):
        pyplot.subplot(312)
        pyplot.title("F1")
        pyplot.plot(epoch_vals, score.train_f1, label='train')
        pyplot.plot(epoch_vals, score.valid_f1, label='valid')
        pyplot.legend()
        pyplot.xticks(epoch_vals)

        pyplot.subplot(313)
        pyplot.title("acc")
        pyplot.plot(epoch_vals, score.train_acc, label='train')
        pyplot.plot(epoch_vals, score.valid_acc, label='valid')
        pyplot.legend()
        pyplot.xticks(epoch_vals)

    pyplot.show()


def prune_preds(y_true, y_pred):
    true = []
    preds = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    for i in range(len(y_true)):
        if(y_true[i] != 0): # ignore padding index 0
            true.append(y_true[i])
            preds.append(y_pred[i])

    return true, preds

