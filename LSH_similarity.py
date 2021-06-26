import pandas as pd
import numpy as np
from datasketch import MinHash
from datasketch import MinHashLSH as lshash
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import sys

def read_sum_data(path):
    sum_database = pd.read_csv(f'./{path}/summary_gpu.csv')
    return sum_database.loc[:,'kobart_sum']

def vectorization(DATABASE,NEWS):
    database =  DATABASE
    news = NEWS

    corpus=[x for x in database]
    corpus.append(news)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X

set1 = set(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'datasets'])
set2 = set(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'documents'])
set3 = set(['minhash', 'is', 'probability', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'documents'])

m1 = MinHash(num_perm=128)
m2 = MinHash(num_perm=128)
m3 = MinHash(num_perm=128)
for d in set1:
    m1.update(d.encode('utf8'))
for d in set2:
    m2.update(d.encode('utf8'))
for d in set3:
    m3.update(d.encode('utf8'))

# Create LSH index
lsh = lshash(threshold=0.5, num_perm=128)
lsh.insert("m2", m2)
lsh.insert("m3", m3)
start = time.time()  # 시작 시간 저장
DATABASE = read_sum_data('./Sum_Database')
NEWS = list(pd.read_csv('./Sum_Database/news.txt'))[0]
print(DATABASE)
sys.exit(0)
X=vectorization(DATABASE,NEWS)
print(X)
sys.exit(0)



result = lsh.query(m1)
print("Approximate neighbours with Jaccard similarity > 0.5", result)