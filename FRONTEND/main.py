import time
import os
import torch
from flask import Flask, render_template, request, redirect, session, url_for, flash
from ast import literal_eval
from itertools import chain
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from transformers import AutoModel, AutoTokenizer, RobertaTokenizerFast, RobertaModel

app = Flask(__name__)
BASE_URL = "../input/nbme-score-clinical-patient-notes"

def create_test_df():
    feats = pd.read_csv(f"{BASE_URL}/features.csv")
    notes = pd.read_csv(f"{BASE_URL}/patient_notes.csv")
    test = pd.read_csv(f"{BASE_URL}/test.csv")

    merged = test.merge(notes, how = "left")
    merged = merged.merge(feats, how = "left")

    def process_feature_text(text):
        return text.replace("-OR-", ";-").replace("-", " ")
    
    merged["feature_text"] = [process_feature_text(x) for x in merged["feature_text"]]
    
    return merged


class SubmissionDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data.loc[idx]
        tokenized = self.tokenizer(
            example["feature_text"],
            example["pn_history"],
            truncation = self.config['truncation'],
            max_length = self.config['max_length'],
            padding = self.config['padding'],
            return_offsets_mapping = self.config['return_offsets_mapping']
        )
        tokenized["sequence_ids"] = tokenized.sequence_ids()

        input_ids = np.array(tokenized["input_ids"])
        attention_mask = np.array(tokenized["attention_mask"])
        offset_mapping = np.array(tokenized["offset_mapping"])
        sequence_ids = np.array(tokenized["sequence_ids"]).astype("float16")

        return input_ids, attention_mask, offset_mapping, sequence_ids


test_df = create_test_df()

hyperparameters = {
    "max_length": 416,
    "padding": "max_length",
    "return_offsets_mapping": True,
    "truncation": "only_second",
    "model_name": "../input/huggingface-roberta-variants/roberta-base/roberta-base",
    "dropout": 0.2,
    "lr": 1e-5,
    "test_size": 0.2,
    "seed": 1268,
    "batch_size": 8
}

tokenizer = RobertaTokenizerFast.from_pretrained(hyperparameters['model_name'])

submission_data = SubmissionDataset(test_df, tokenizer, hyperparameters)
submission_dataloader = DataLoader(submission_data, batch_size=hyperparameters['batch_size'], shuffle=False)

# model.eval()
preds = []
offsets = []
seq_ids = []


#preds = np.concatenate(preds, axis=0)
#offsets = np.concatenate(offsets, axis=0)
#seq_ids = np.concatenate(seq_ids, axis=0)


# Routes

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('landing.html')

        
@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'GET':
        return render_template('test.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)

