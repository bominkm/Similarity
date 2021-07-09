import pandas as pd
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import numpy as np

import torch
import time

def read_sum_data(path):
    sum_database = pd.read_csv(path)
    return sum_database.loc[:,'kobart_sum']

def vectorization(DATABASE,NEWS,type):
    database =  DATABASE
    news = NEWS
    corpus=[x for x in database]

    if type =='tfidf':
        corpus.append(news)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        
    if type =='bert':
        model_path = './KoSentenceBERT_SKTBERT/output/training_sts'
        embedder = SentenceTransformer(model_path)
        query = NEWS

        corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        print(corpus_embeddings)
        print(query_embedding)
        X = corpus_embeddings + query_embedding

    return X


def Cosine_similarity(X,DATABASE,NEWS,type,top_k):
    news = X[-1]
    laws = X[:-1]
    database=DATABASE
    if type == 'tfidf':
        cos_scores = cosine_similarity(news,laws)[0]
        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

        print("\n\n======================\n\n")
        print("Query:", NEWS)
        print(f"\nTop {top_k} most similar sentences in corpus:")

        for idx in top_results[0:top_k]:
            print(database[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))

        result = pd.DataFrame({'corpus':database,
                                'similarity':cos_scores})
        final = result.sort_values('similarity').iloc[:top_k,0]
    
    if type =='bert':
        cos_scores = util.pytorch_cos_sim(news, laws)[0]
        cos_scores = cos_scores.cpu()
        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

        print("\n\n======================\n\n")
        print("Query:", news)
        print(f"\nTop {top_k} most similar sentences in corpus:")

        for idx in top_results[0:top_k]:
            print(database[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))

        result = pd.DataFrame({'corpus':database,
                                'similarity':cos_scores})
        final = result.sort_values('similarity').iloc[:top_k,0]

    return final

def make_simtext(law_path,new_path,type,top_k):
    DATABASE = read_sum_data(law_path)
    NEWS = list(pd.read_csv(new_path))[0]
    X=vectorization(DATABASE,NEWS,type)
    final = Cosine_similarity(X,DATABASE,NEWS,type,top_k)

    for i,f in enumerate(final):
        text_file = open(f"./Output/output_{i+1}.txt", "w")
        n = text_file.write(f)
        text_file.close()

if __name__ == '__main__':
    #start = time.time()  # 시작 시간 저장
    parser = argparse.ArgumentParser()
    # Directory paths
    parser.add_argument('--law-dir', default='./Sum_Database/summary_gpu.csv')
    parser.add_argument('--news-dir', default='./Sum_Database/news.txt')
    parser.add_argument('--emb-type', required=True, choices=['tfidf','fasttext','bert'])
    parser.add_argument('--top-k',default = 3)
    args = parser.parse_args()

    make_simtext(args.law_dir,args.news_dir,args.emb_type,args.top_k)
         
    #print("time :", time.time() - start)




