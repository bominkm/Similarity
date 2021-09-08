# 뉴스 유사 판례 제공 웹서비스
## 1. About

<p align="center"><img src="https://github.com/bominkm/Similarity/blob/master/PIN.png?raw=true"></p>

- **제 14회 빅데이터 연합동아리 BOAZ 빅데이터 컨퍼런스 프로젝트**입니다.  
- **PIN 서비스**는 Precedents In News의 약어로, **뉴스와 유사한 판례를 제공하는 웹서비스**입니다.

<br>

## 2. Flow Chart
<p align="center"><img src="https://github.com/bominkm/Similarity/blob/master/Flow-Chart-Modeling.png?raw=true"></p>
<p align="center"><img src="https://github.com/bominkm/Similarity/blob/master/Flow-Chart-DB.png?raw=true"></p>

<br>

### How to calculate Cosine Similarity 

* input1 : Sum_Dataset/[판례데이터].csv [law]
* input2 : Sum_Dataset/[입력뉴스].txt [input news]
* output : Output/output_1.txt ~ output_3.txt [Cosine similarity top 3 law original corpus]

### How to Use Bert
1. concat.csv, embeddings.pkl [다운로드](https://drive.google.com/drive/u/0/folders/14j8_1jkGr3KwI88Xl53k9J63kqID9L16)
2. /Sum_Database 경로에 넣기

<br>

## 3. Result
<p align="center"><img src="https://github.com/bominkm/Similarity/blob/master/Result.png?raw=true"></p>

<br>
