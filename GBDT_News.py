#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:10:48 2019

@author: sivan
"""

import os
import sys
import regex as re
from sklearn.model_selection import train_test_split
from sklearn import datasets
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
nltk.data.path.append("/Users/sivan/nltk_data")

#from sklearn.feature_extraction.text import TfidfVectorizer

'''读取文件信息 采用sklearn来进行数据读取'''
raw_data=datasets.load_files('20_newsgroups',encoding='latin-1')
#加载该目录下面的所有文件
print(len(raw_data.filenames))
#获取到加载到的数据的数据集
X=raw_data.data
print(X[0])
#获取加载到的数据的类别标签数据，已经将其转化为数值型
Y=raw_data.target
print(Y[4867])
#获取类别标签的具体含义，即每个子文件夹的名字
target_names=raw_data.target_names
print(target_names)

'''读取文件数据 使用os来进行文件的读取'''
# =============================================================================
# TEXT_DATA_DIR="../20_newsgroups/"
# texts = []  # 存放文本数据
# labels_index = {}  # 字典，标签和标签id进行对应
# labels = []  # 存放标签id
# for name in sorted(os.listdir(TEXT_DATA_DIR)):
#     path = os.path.join(TEXT_DATA_DIR, name)
#     if os.path.isdir(path):
#         label_id = len(labels_index)
#         labels_index[name] = label_id
#         for fname in sorted(os.listdir(path)):
#             if fname.isdigit():
#                 fpath = os.path.join(path, fname)
#                 if sys.version_info < (3,):
#                     f = open(fpath)
#                 else:
#                     f = open(fpath, encoding='latin-1')
#                 data=f.read()
#                 if data in texts:
#                     re_index=texts.index(data)
#                     texts.pop(re_index)
#                     labels.pop(re_index)
#                     f.close()
#                 else:
#                     texts.append(data)  # 整篇文章读取，不是按行读取
#                     f.close()
#                     labels.append(label_id)
# =============================================================================


'''清洗数据 处理大小写、数字、空格、标点'''
def clean_text(text):
    text=text.lower()
    text = re.sub("\d+", " ", text) # 去除数字
    text = re.sub("\p{P}+", " ", text) # 去除标点符号
    text = re.sub("<", " ", text)
    text = re.sub(">", " ", text)
    text = re.sub(r'\s+', " ", text) # 多个空格合并一个空格   
    return text
 
'''使用停用词'''
def remove_stopwords(text):
    stopworddic = set(stopwords.words('english'))
    text = [i for i in text if i not in stopworddic ]
    text=' '.join(text)
    return text
 
'''词干提取'''
def stemmed_word(word):
    porter_stemmer = PorterStemmer()
    return porter_stemmer.stem(word)

def stemmed_text(text):
    text = [stemmed_word(word) for word in text.split(" ")]
    text = ' '.join(text)
    return text
 
i=0
length=len(X)
datas=X
while i < length:
    X[i]= stemmed_text(clean_text(X[i]))
    datas[i]=word_tokenize(X[i])
    datas[i]=remove_stopwords(datas[i])
    i+=1
 
'''将数据集划分为训练集和测试集'''
X_train, X_test, y_train, y_test =train_test_split(datas, Y, test_size=0.3, random_state=0) 
 
'''数据进行分词操作'''
count_vec=CountVectorizer()
# 使用词频统计的方式将原始训练和测试文本转化为特征向量
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)
transformer = TfidfTransformer(norm='l2',smooth_idf=True)
X_train_tfidf=transformer.fit_transform(X_count_train)
X_test_tfidf=transformer.transform(X_count_test)
 
'''GBDT分类'''
gb_model = GradientBoostingClassifier(n_estimators=10)
gb_model.fit(X_train_tfidf, y_train)
# 将分类预测的结果存储在变量gb_model_predict中
y_count_predict = gb_model.predict(X_test_tfidf)
print(accuracy_score(y_count_predict, y_test))
print(classification_report(y_test, y_count_predict, target_names = target_names))
 

#'''贝叶斯分类'''
#nb_model=MultinomialNB()
#nb_model.fit(X_train_tfidf,y_train)
#y_count_predict = nb_model.predict(X_test_tfidf)
#print(accuracy_score(y_count_predict, y_test))
#print(classification_report(y_test, y_count_predict, target_names =labels_index.keys()))

#'''SVM分类'''
#svm_model=SVC(C=0.1,gamma=0.7,kernel="linear")
#svm_model.fit(X_train_tfidf, y_train)
#y_count_predict = svm_model.predict(X_test_tfidf)
#print(classification_report(y_test, y_count_predict, target_names =labels_index.keys()))



