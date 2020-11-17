class Config:
    # data_path = "/home/ehoang/git/python/tweet_classification/data/CrisisLexT26/2013_Australia_bushfire/2013_Australia_bushfire-tweets_labeled.csv"
    data_path = "/home/ehoang/git/python/tweet_classification/data/COVID19Tweet/data.csv"
    # selected_columns = ['Tweet ID', ' Tweet Text', ' Informativeness']
    selected_columns = ['id', 'tweet', 'label']
    columns = ['id', 'tweet', 'label']
    ignored_labels = ['Not related', 'Not applicable']
    summary_len = 20
    # nonInforLabel = "Related - but not informative"
    nonInforLabel = "UNINFORMATIVE"

    TFIDF = True 
    BERT_SCORE = True 
    BERT_FIRST_TOKEN = False 

    bert_score_in = "/home/ehoang/hnt/tweet_summarization/data/inputs/bert_score_"
    bert_score_out = "/home/ehoang/hnt/tweet_summarization/data/outputs/"
    tfidf_out= "/home/ehoang/hnt/tweet_summarization/data/outputs/tfidf_"
