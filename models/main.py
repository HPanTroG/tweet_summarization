import os
import sys
sys.path.append(os.path.dirname("/home/ehoang/hnt/tweet_summarization/"))
import numpy as np
import pandas as pd
from utils import Lexrank, tokenizeRawTweetText
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re, emoji, string
from bert_score import BERTScorer
from joblib import Parallel, delayed
from config.config import Config


def process_data(data):
    data['tweet_pro'] = data['tweet'].apply(lambda x: ' '.join(tokenizeRawTweetText(x)))
    # # remove rt, @USER, @URL, emoji
    data['tweet_pro'] = data['tweet_pro'].apply(lambda x: x.replace('TWEETMENTION', "").
                                        replace("EMAILADDRESS", "").replace('HTTPURL', ''))
    data['tweet_pro'] = data['tweet_pro'].apply(lambda x: x.lower().strip())
    data['tweet_pro'] = data['tweet_pro'].apply(lambda x: re.sub("( rt)|(rt )", "", x))                              
    data['tweet_pro'] = data['tweet_pro'].apply(lambda x: re.sub('^( ?: ?)', '', x))
    data['tweet_pro'] = data['tweet_pro'].apply(lambda x: re.sub("  +", " ", x))
    data['tweet_pro'] = data['tweet_pro'].apply(lambda x: ''.join(c for c in x if c not in emoji.UNICODE_EMOJI).strip())

    # remove stopwords, punctuation
    stopWords = stopwords.words('english')
    data['tweet_clean'] = data['tweet_pro'].apply(lambda x: ' '.join(y for y in x.split(" ") if y not in stopWords))

    data['tweet_clean'] = data['tweet_clean'].apply(lambda x: x.translate(str.maketrans('', '',  string.punctuation)))
    data['tweet_clean'] = data['tweet_clean'].apply(lambda x: re.sub('“|…|’|‘|”|—|→', "", x))
    data['tweet_clean'] = data['tweet_clean'].apply(lambda x: re.sub(' +', ' ',x).strip())

    # remove tweets #unique words less than haft of length
    data['uniWPercent'] = data['tweet_clean'].apply(lambda x: 0 if len(set(x.split(" ")))/len(x.split(" ")) <= 0.5 else len(x.split(" ")))
    data = data[data['uniWPercent']!=0]
    # # remove tweets with lengths < 3, duplicates
    while data['uniWPercent'].min() <=2:
        data = data[data['uniWPercent'] >2]
        data['uniWPercent'] = data['tweet_clean'].apply(lambda x: 0 if len(set(x.split(" ")))/len(x.split(" ")) <= 0.5 else len(x.split(" ")))
    # # # # remove duplicates
    data = data.reset_index(drop=True)
    # data.drop_duplicates(subset=['tweet_clean'], keep='first', inplace = True)
    # remained_index = data.index
    # data = data.reset_index(drop=True)
    return data

def extract_bertscore(data, column, batch, thres, output_path):
   
    scorer = BERTScorer(lang='en', rescale_with_baseline = True, idf = True, idf_sents = list(data[column]), 
                               device = 'cuda')
    # scorer = BERTScorer(lang='en', rescale_with_baseline = True, idf = False, 
    #                            device = 'cuda')
    # print("device: {}..running {}".format(device, len(batch)))
    
    for idx in batch:
        # compute bert score
        batch_size = 1000
        sim_score = []
        sim_idx = []
        for i in range(idx+1, data.shape[0], batch_size):
            rightBound = i+batch_size
            if i + batch_size > data.shape[0]:
                rightBound = data.shape[0]
            sim = scorer.score([str(data.iloc[idx][column])]*(rightBound -i), list(data.iloc[i:rightBound][column]))[0]
            sim_score+=list(sim.numpy())
        sim_score = np.array(sim_score)
        sim_idx = np.where(sim_score>thres)[0]
        sim_score = sim_score[sim_idx]
        if (len(sim_idx) !=0):
            sim_idx = sim_idx+(idx+1)
        with open(output_path, 'a') as f:
            f.write("{},{},{}\n".format(idx, str(list(sim_idx)), str(list(sim_score))))


if __name__ == "__main__":
    print("Read data ...")
    data = pd.read_csv(Config.data_path)
    data = data[Config.selected_columns]
    data.columns = Config.columns
    # remove non-relevant tweets
    data = data[~data['label'].isin(Config.ignored_labels)]
    print(data.shape)
    print(Config.data_path[Config.data_path.rindex('/')+1:-4])
    print("Clean data ...")
    data = process_data(data)
    for i in range(5):
        print(i, ".....")
        print(str(data.iloc[i]['tweet']))
        print(str(data.iloc[i]['tweet_pro']))
        print(str(data.iloc[i]['tweet_clean']))
    print(data.shape)
    print("...............................................")

    if Config.TFIDF == True:
        print("............Tfidf+Lexrank.............")
        tfidf = TfidfVectorizer()
        tfidfData = tfidf.fit_transform(data['tweet_clean'])
        print(tfidfData.shape)
        lex_tfidf = Lexrank(tfidfData)
        lex_tfidf.build_graph_cosine(cos_thres=0.3, batch_size=1000)
        lex_tfidf.train(lexrank_iter=100, damping_factor=0.85)
        sentIds= lex_tfidf.extract_summary(n_sents=Config.summary_len, cos_thres=0.35, max_sent=200)
        print("Summary: ")
        print(sentIds)
        output_path = Config.tfidf_out+ Config.data_path[Config.data_path.rindex('/')+1: -4]+".txt"
        print(output_path)
        count_flase_return = 0
        with open(output_path, "w") as f:
            for i, idx in enumerate(sentIds):
                if data.iloc[idx]['label'] == Config.nonInforLabel:
                    count_flase_return+=1
                f.write("{}. {} {}\t{}\n".format(i+1, lex_tfidf.scores[idx], data.iloc[idx]['tweet_clean'], data.iloc[idx]['label']))
            f.write("..........Uncleaned data............\n")
            for i, idx in enumerate(sentIds):
                f.write("{}. {} {}\n".format(i+1, lex_tfidf.scores[idx], data.iloc[idx]['tweet']))
                print("{}. {} {} {}".format(i+1, lex_tfidf.scores[idx], data.iloc[idx]['tweet'], data.iloc[idx]['label']))
        print("Precision: ", 1-count_flase_return/Config.summary_len)
    if Config.BERT_SCORE == True:
        input_file = Config.bert_score_in + Config.data_path[Config.data_path.rindex('/')+1:-4]+".txt"
        print(input_file)

        if os.path.exists(input_file):
            lex = Lexrank(data)
            lex.build_graph_bertscore_from_file(sim_thres=0.15, input_file=input_file)
            lex.train(lexrank_iter=100, damping_factor=0.85)
            bertscore_dict = {}
            for key, value in lex.graph.items():
                bertscore_dict[key] = len(value)
            bertscore_dict = {k: v for k, v in sorted(bertscore_dict.items(), key=lambda item: item[1], reverse = True)}
            count = 0
            selected = []
            print("...........................................")
            count_flase_return = 0
            for key, value in bertscore_dict.items():
                
                if count>0:
                    added = True
                    for k in selected:
                        if k in lex.graph[key]:
                            if lex.graph[key][k]>1:
                                added = False
                                break
                    
                    if added==True:
                        if data.iloc[key]['label'] == Config.nonInforLabel:
                            count_flase_return+=1
                        selected.append(key)
                        count+=1
                        print(count, ".", key, len(lex.graph[key]), str(data.iloc[key]['tweet']), data.iloc[key]['label'])
                else:
                    selected.append(key)
                    if data.iloc[key]['label'] == Config.nonInforLabel:
                            count_flase_return+=1
                    count+=1
                    print(count, ".", key, len(lex.graph[key]), str(data.iloc[key]['tweet']), data.iloc[key]['label'])
                        
                if count >= Config.summary_len: 
                    break
            print("Precision: ", 1-count_flase_return/Config.summary_len)
            print("...................................................")
            sentIds = lex.extract_summary(n_sents=Config.summary_len, cos_thres=0.25, 
                        max_sent=data.shape[0])
            count_flase_return = 0
            output_path = Config.bert_score_out+ input_file[input_file.rindex('/')+1: -4]+".txt"
            with open(output_path, "w") as f:
                for i, idx in enumerate(sentIds):
                    if data.iloc[idx]['label'] == Config.nonInforLabel:
                        count_flase_return +=1
                    f.write("{}. {} {}   {}\n".format(i+1, len(lex.graph[idx]), data.iloc[idx]['tweet_pro'], data.iloc[idx]['label']))
                f.write("..........Uncleaned data............\n")
                for i, idx in enumerate(sentIds):
                    f.write("{}. {} {}\n".format(i+1, lex.scores[idx], data.iloc[idx]['tweet']))
            print("Precision: ", 1- count_flase_return/Config.summary_len)

        else:
            print("Extracting bert score...")
            
            extract_bertscore(data, 'tweet_pro', np.arange(0, data.shape[0]-1, 1), 0.0, input_file)


                
