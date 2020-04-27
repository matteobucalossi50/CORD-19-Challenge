#!/usr/bin/env python
# coding: utf-8

import json
import glob
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np

from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

dirbase='/content/drive/My Drive/Colab Notebooks/final/CORD-19-research-challenge/'


#Fetch All of JSON File Path
json_path = glob.glob(f'{dirbase}/**/*.json', recursive=True)[0:5000]
len(json_path)

with open(json_path[0]) as file:
    content = json.load(file)

class FileReader:
    def __init__(self, dirbase):
        with open(dirbase) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract}... {self.body_text}...'
first_file = FileReader(json_path[0])
print(first_file)


def read_directory_files(path):
    file_texts = {'paper_id':[],'abstract':[],'body_text':[]}
    for i in range(0,len(json_path)):
        input_file_text = open(json_path[i])
        with input_file_text as f:
            data = json.load(f)
        file_texts['paper_id'].append(data['paper_id'])
        if 'abstract' not in data.keys():
            file_texts['abstract'].append(None)
        elif data['abstract']==[]:
            file_texts['abstract'].append(None)
        else:
            file_texts['abstract'].append(data['abstract'])

        if 'body_text' not in data.keys():
            file_texts['body_text'].append(None)
        elif data['body_text']==[]:
            file_texts['body_text'].append(None)
        else:
            file_texts['body_text'].append(data['body_text'])


    df=pd.DataFrame(data=file_texts)
    return df


# may run a little bit longer
json_text = read_directory_files(json_path)

json_text.head()

print(list(json_text['body_text'][0][0]['text'].split()))


json_text['body_text'].head()


print(json_text.shape)



# import necessary libraries
import nltk.data
import nltk
nltk.download('punkt')
import re
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
treebank_tokenizer = TreebankWordTokenizer()
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
from nltk.stem.porter import *
porter_stemmer = PorterStemmer()


# TF-IDF Vectorization
# (silly way) REPLACED

type(word_tokenize(json_text['body_text'][0][0]['text']))

value=[]
for i in range(0,len(json_text['body_text'])):
  if json_text['body_text'][i] is None:
    value.append('NA')
  elif json_text['body_text'][i][0]['text'] is None:
    value.append('NA')
  else:
    value.append(word_tokenize(json_text['body_text'][i][0]['text']))
#value[0:3]
json_text['preprocessing_bodytext']=value

print(json_text.shape)


json_text['preprocessing_bodytext'].head()

type(json_text['preprocessing_bodytext'][0][0])


from sklearn.feature_extraction.text import TfidfVectorizer

def Encode(data):
    #create the transform
    missing_value=True
    value=[]
    for i in data:
        if i is None:
            continue
        else:
            for i2 in i:
                if i2['text'] is None:
                  value.append(None)
                else:
                  value.append(list(i2['text'].split()))

    return value


body_text=Encode(json_text['body_text'])
for i in body_text[0:2]:
  print(i)
len(body_text)


from sklearn.feature_extraction.text import TfidfVectorizer

def EncodeTFIDF(data, maxfeature):
  try:
    value=[]
    #text=list(data['text'].split())
    for i in range(0,len(data)):
      m=','.join(data[i])
      value.append(m)
    vectorizer=TfidfVectorizer(stop_words='english',use_idf=True,max_features=maxfeature)
  #vectorizer.fit(data)
    value = vectorizer.fit_transform(value)
    #word=vectorizer.get_feature_names()
    #print(word)
  except ValueError:
    pass
  return value


bodytext_vect=EncodeTFIDF(json_text['preprocessing_bodytext'].values,2**10)


print(bodytext_vect.shape)

#bodytext_vect=bodytext_vect.toarray



##############################################################################
################### models and algorithms to follow ##########################
##############################################################################


##############################################################################
# Dimensions Reduction using UAMP

get_ipython().system('pip install umap')

import umap.umap_ as umap
import matplotlib.pyplot as plt


reducer = umap.UMAP(n_neighbors = 5)


clusterable_embedding = reducer.fit_transform(bodytext_vect.toarray())
plt.figure(figsize=(12,8))
plt.scatter(clusterable_embedding[:,0],clusterable_embedding[:,1])
print(clusterable_embedding.shape)
print(clusterable_embedding)


##############################################################################
# Clustering using HDBSCAN

import hdbscan
import numpy as np
import seaborn as sns
import pandas as pd


#clusterable_embedding

clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
clusterer = clusterer.fit(clusterable_embedding)

#Build the minimum spanning tree
clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)

#Build the cluster hierarchy
clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)

#Condense the cluster tree
clusterer.condensed_tree_.plot()

#Extract the clusters
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())

#palette = sns.color_palette()
#cluster_colors = [sns.desaturate(palette[col], sat)
#                  if col >= 0  else (0.5, 0.5, 0.5) for col, sat in
#                  zip(clusterer.labels_, clusterer.probabilities_)]
#plt.scatter(clusterable_embedding.T[0], clusterable_embedding.T[1], c=cluster_colors)


#clusterer = hdbscan.HDBSCAN(min_cluster_size=10, prediction_data=True).fit(clusterable_embedding)
color_palette = sns.color_palette('Paired',max(clusterer.labels_))
cluster_colors = [color_palette[x] if x >= 0 and x<max(clusterer.labels_)
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in
                        zip(cluster_colors, clusterer.probabilities_)]
plt.scatter(*clusterable_embedding.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)


print(clusterable_embedding.T[0])
print(clusterable_embedding.T[1])

print(clusterer.labels_)

len(clusterer.labels_)

json_text['cluster_textbody'] = clusterer.labels_

max(clusterer.labels_)

max(json_text['cluster_textbody'])

grouped=json_text.groupby('cluster_abstract')
for gp_name, gp in grouped:
    display(gp)



##############################################################################
# Topic Modeling on Each Cluster using LDA

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


count_vectorizer = CountVectorizer(stop_words='english')
#Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(corona_dataframe['Text'])
#Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)


def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

#Tweak the two parameters below
number_topics = 5
number_words = 10

#Create and fit the LDA model imported from sklearn library
lda = LDA(n_components=number_topics, n_jobs=1)
lda.fit(count_data)

#Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)


lda_bodytext = LatentDirichletAllocation(n_components=50, random_state=0)
lda_bodytext.fit(bodytext_vect)


tfidf_feature_names = clusterable_embedding.get_feature_names()

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "\nTopic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

print_top_words(lda_tf, tfidf_feature_names, 25)


abstract = json_text['abstract']
abstract.fillna("",inplace=True)


lda = LatentDirichletAllocation().fit(bodytext_vect)


def display_topics(model, feature_names, no_top_words):
    topics=[]
    for topic_idx, topic in enumerate(model.components_):
        #rint ("Topic %d:" % (topic_idx))
        topic_words=" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        #rint(topic_words)
        topics.append(topic_words)
    return topics


no_top_words = 5

topics_lda=display_topics(lda, tf_feature_names, no_top_words)

pred_lda=lda.transform(X_tf)

res_lda=[topics_lda[np.argmax(r)] for r in pred_lda]

corona_df['topic_lda']=res_lda

grouped=corona_df.groupby('topic_lda')
for gp_name, gp in grouped:
    display(gp)

vectorizers = []

for ii in range(0, 20):
    # Creating a vectorizer
    vectorizers.append(CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}'))


# vectorize the data from each cluster
vectorized_data = []

for current_cluster, cvec in enumerate(vectorizers):
    try:
        vectorized_data.append(cvec.fit_transform(df.loc[df['y'] == current_cluster, 'processed_text']))
    except Exception as e:
        print("Not enough instances in cluster: " + str(current_cluster))
        vectorized_data.append(None)


len(vectorized_data)


# number of topics per cluster
NUM_TOPICS_PER_CLUSTER = 20

lda_models = []
for ii in range(0, 20):
    # Latent Dirichlet Allocation Model
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS_PER_CLUSTER, max_iter=10, learning_method='online',verbose=False, random_state=42)
    lda_models.append(lda)

print(lda_models[0])


clusters_lda_data = []

for current_cluster, lda in enumerate(lda_models):
    # print("Current Cluster: " + str(current_cluster))

    if vectorized_data[current_cluster] != None:
        clusters_lda_data.append((lda.fit_transform(vectorized_data[current_cluster])))


def selected_topics(model, vectorizer, top_n=3):
    current_words = []
    keywords = []

    for idx, topic in enumerate(model.components_):
        words = [(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])

    keywords.sort(key = lambda x: x[1])
    keywords.reverse()
    return_values = []
    for ii in keywords:
        return_values.append(ii[0])
    return return_values


all_keywords = []
for current_vectorizer, lda in enumerate(lda_models):
    # print("Current Cluster: " + str(current_vectorizer))

    if vectorized_data[current_vectorizer] != None:
        all_keywords.append(selected_topics(lda, vectorizers[current_vectorizer]))

print(all_keywords[0][:10])

