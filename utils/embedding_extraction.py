import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import torch
from scipy.sparse import coo_matrix, vstack, save_npz

def get_bert_first_token_embeddings(data, cuda="cuda:3", dataColumn="Tweet", max_len=60):
    """
        Extract embeddings of first token using Bert model
        Input: data -- dataframe
        Output: embeddings -- shape: (data.shape[0], 768)
    """
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device(cuda)
    model.to(device)
    data['len(tokenize)'] = data['Tweet'].apply(lambda x: len(tokenizer.tokenize(x)))
    max_len = data['len(tokenize)'].quantile(0.99)
    print("Max_len (99% data): {}".format(max_len))

    encoded_data = data[dataColumn].apply(lambda x: tokenizer.encode_plus(
        x, pad_to_max_length=True, max_length=int(max_len), truncation = True
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
            input_ids = list(encoded_data[leftBound: rightBound]['input_ids'])

            features = model(torch.tensor(input_ids).to(device))[0][:, 0, :].to('cpu')
            if i == 0:
                embeddings = features
            else:
                embeddings = np.concatenate((embeddings, features), axis = 0)
            print(embeddings.shape)

            leftBound += batch_size
            i += 1
    return embeddings


def get_bert_all_token_embeddings(data, cuda="cuda:3", dataColumn="Tweet", max_len=60, file=""):
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device(cuda)
    model.to(device)
    
    data['len(tokenize)'] = data['Tweet'].apply(lambda x: len(tokenizer.tokenize(x)))
    max_len = data['len(tokenize)'].quantile(0.99)
    print("Max_len (99% data): {}".format(max_len))

    encoded_data = data[dataColumn].apply(lambda x: tokenizer.encode_plus(
        x, pad_to_max_length=True, max_length=int(max_len), truncation=True
    ))
    encoded_data = pd.DataFrame(encoded_data.values.tolist())
    print("Encoded data: \n{}".format(encoded_data.head()))

    leftBound = 0
    batch_size = 1000
    embeddings = None
    i = 0
    sparse_matrix = coo_matrix((0, 0))
    with torch.no_grad():
        while leftBound < encoded_data.shape[0]:
            rightBound = leftBound + batch_size
            if rightBound > encoded_data.shape[0]:
                rightBound = encoded_data.shape[0]
#             print("----------", leftBound, rightBound)
            input_ids = encoded_data[leftBound: rightBound]['input_ids'].array
           
            attention_masks = np.array(list(encoded_data[leftBound: rightBound]['attention_mask']))
            
            n0 = attention_masks.shape[0]
            n1 = attention_masks.shape[1]
            attention_masks = attention_masks.reshape(n0, n1, 1)
            
            features = model(torch.tensor(input_ids).to(device))[0].to('cpu')
            
            # zero all embedding of padded elements
            features = features * attention_masks
            
            features = features[:, 1:-1, :].reshape(features.shape[0], -1)
            if sparse_matrix.shape[0] == 0:
                sparse_matrix = coo_matrix(features)
                
            else:
                
                sparse_matrix = vstack([sparse_matrix, coo_matrix(features)])
#             print(sparse_matrix.shape)
            
            #save file
            if (rightBound%4000 == 0) or (rightBound == encoded_data.shape[0]):
                print("Matrix: {}, #Non-zeros: {}".format(sparse_matrix.shape, sparse_matrix.getnnz()))
                output = file+str(rightBound/4000)+".npz"
                print("File: ", output)
                save_npz(output, sparse_matrix)
                sparse_matrix = coo_matrix((0, 0))
                
            leftBound += batch_size
            i += 1
    


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
            print("Len: ", embeddings.shape)

    return embeddings
