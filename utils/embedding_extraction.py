import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import torch


def get_bert_first_token_embeddings(data, cuda="cuda:3", dataColumn="Tweet", max_len=60):
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device(cuda)
    model.to(device)

    encoded_data = data[dataColumn].apply(lambda x: tokenizer.encode_plus(
        x, pad_to_max_length=True, max_length=60, truncation = True
    ))
    encoded_data = pd.DataFrame(encoded_data.values.tolist())
    print("Encoded data: \n{}".format(encoded_data.head()))

    leftBound = 0
    batch_size = 1000
    embeddings = None
    i = 0
    with torch.no_grad():
        while leftBound < encoded_data.shape[0]:
            rightBound = leftBound + batch_size
            if rightBound > encoded_data.shape[0]:
                rightBound = encoded_data.shape[0]
            print("----------", leftBound, rightBound)
            input_ids = encoded_data[leftBound: rightBound]['input_ids'].array

            features = model(torch.tensor(input_ids).to(device))[0][:, 0, :]
            if i == 0:
                embeddings = features
            else:
                embeddings = torch.cat((embeddings, features), 0)
            print(embeddings.shape)

            leftBound += batch_size
            i += 1
    return embeddings


def get_bert_all_token_embeddings(data, cuda="cuda:3", dataColumn="Tweet", max_len=60):
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device(cuda)
    model.to(device)

    encoded_data = data[dataColumn].apply(lambda x: tokenizer.encode_plus(
        x, pad_to_max_length=True, max_length=60, truncation=True
    ))
    encoded_data = pd.DataFrame(encoded_data.values.tolist())
    print("Encoded data: \n{}".format(encoded_data.head()))

    leftBound = 0
    batch_size = 1000
    embeddings = None
    i = 0
    with torch.no_grad():
        while leftBound < encoded_data.shape[0]:
            rightBound = leftBound + batch_size
            if rightBound > encoded_data.shape[0]:
                rightBound = encoded_data.shape[0]
            print("----------", leftBound, rightBound)
            input_ids = encoded_data[leftBound: rightBound]['input_ids'].array
            attention_masks = encoded_data[leftBound: rightBound]['attention_mask'].array
            features = model(torch.tensor(input_ids).to(device))[0]
            features = features * attention_masks
            features = features[:, 1:-1, :]
            if i == 0:
                embeddings = features
            else:
                embeddings = torch.cat((embeddings, features), 0)
            print(embeddings.shape)

            leftBound += batch_size
            i += 1
    return embeddings


def get_sentence_transformers_embedings(data, cuda="cuda:3", dataColumn="Tweet", max_len=60):
    modelSent = SentenceTransformer('bert-base-nli-mean-tokens')

    leftBound = 0
    batch_size = 1000
    embeddings = np.empty((0, 768))

    with torch.no_grad():
        while  leftBound < data.shape[0]:
            rightBound = leftBound+batch_size
            if rightBound > data.shape[0]:
                rightBound = data.shape[0]
            embeddings = np.concatenate((embeddings, modelSent.encode(list(data.iloc[leftBound:rightBound][dataColumn]))), axis=0)
            leftBound += batch_size
            print("Len: ", len(embeddings))

    return embeddings
