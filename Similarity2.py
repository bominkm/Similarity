import pandas as pd
import numpy as np
import torch
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import time

def read_law_data(path):
    sum_database = pd.read_csv(path)
    sum_database=sum_database[sum_database.kobart_sum.notna()]
    return sum_database.kobart_sum.values.tolist()

def read_news_data(path):
    news_data = list(pd.read_csv(path))[0]
    string = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]"," ", news_data)
    string = re.sub(r'[·@%\\*=()/~#&\+á‘’“”?\xc3\xa1\-\|\:\;\!\-\,\_\~\$\'\"\[\]]', ' ', string) #remove punctuation
    string = re.sub(r'\n',' ', string)     # remove enter
    string = re.sub(r'[0-9]+', '', string) # remove number
    string = re.sub(r'\s+', ' ', string)   #remove extra space
    cleaned_news_data = re.sub(r'<[^>]+>',' ',string) #remove Html tags
    return cleaned_news_data

def vectorization(DATABASE,NEWS):
    model_path = './-2021-07-14_21-48-16'
    embedder = SentenceTransformer(model_path)
    query = NEWS

    #corpus_embeddings = torch.load('./Sum_Database/law_encoder.pt')
    if torch.cuda.is_available():
        with open('./Sum_Database/embeddings.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            #corpus_sentences = stored_data['sentences']
            corpus_embeddings = stored_data['embeddings']
    else:
        with open('./Sum_Database/embeddings_cpu.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            #corpus_sentences = stored_data['sentences']
            corpus_embeddings = stored_data['embeddings']
        

    query_embedding = embedder.encode(query, convert_to_tensor=True)
    X = (corpus_embeddings,query_embedding)
    return X

def Cosine_similarity(X,DATABASE):
    top_k = 3 ; threshhold=0.3
    news = X[1]
    laws = X[0]
    cos_scores = util.pytorch_cos_sim(news, laws)[0]
    cos_scores = cos_scores.cpu()

    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    final=[]
    for idx in top_results[0:top_k]:
        if cos_scores[idx] < threshhold:
            final.append('유사한 판례가 없습니다.')
        else:
            final.append(DATABASE[idx].strip())
    return final

def make_simtext(law_path,news_path):
    DATABASE = read_law_data(law_path)
    NEWS = read_news_data(news_path)
    X=vectorization(DATABASE,NEWS)
    final = Cosine_similarity(X,DATABASE)

    for i,f in enumerate(final):
        text_file = open(f"./Output/output_{i+1}.txt", "w")
        n = text_file.write(f)
        text_file.close()

if __name__ == '__main__':
    #start = time.time()  # 시작 시간 저장
    law_path = './Sum_Database/concat.csv'
    news_path = './Sum_Database/news.txt'
    make_simtext(law_path,news_path)
         
    #print("time :", time.time() - start)
