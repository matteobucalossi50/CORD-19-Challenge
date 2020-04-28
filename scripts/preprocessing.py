import numpy as np
import pandas as pd
import os
import glob
import json
#import scispacy
import spacy
import tqdm
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#nlp = spacy.load("en_core_sci_lg")

import nltk.data
import re
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
treebank_tokenizer = TreebankWordTokenizer()
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
from nltk.stem.porter import *
porter_stemmer = PorterStemmer()


# directories and paths
root_path = os.getcwd()
# metadata_path = f'{root_path}/all_sources_metadata_2020-03-13.csv'
# metadata = pd.read_csv(metadata_path)
# metadata.head()
# metadata.info()

all_json = glob.glob(root_path+'/**/*.json', recursive=True)   # here imo it's where shit happens
print(len(all_json))

# json reader calss
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
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
        return f'{self.paper_id}: {self.abstract}... {self.body_text[:500]}...'
# first_row = FileReader(all_json[0])
# print(first_row)


# # dataframe creation
# def read_directory_files(path):
#     dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': []}
#     for idx, entry in enumerate(path):
#         content = FileReader(entry)
#
#         # get metadata information
#         meta_data = metadata.loc[metadata['sha'] == content.paper_id]
#
#         # no metadata, skip this paper
#         if len(meta_data) == 0:
#             continue
#
#         dict_['paper_id'].append(content.paper_id)
#         dict_['abstract'].append(content.abstract)
#         dict_['body_text'].append(content.body_text)
#
#         # get metadata information
#         meta_data = metadata.loc[metadata['sha'] == content.paper_id]
#
#         # add the authors
#         dict_['authors'].append(meta_data['authors'].values[0])
#
#         # add the title
#         dict_['title'].append(meta_data['title'].values[0])
#
#         # add the journal
#         dict_['journal'].append(meta_data['journal'].values[0])
#
#     df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal'])
#     return df_covid
#
# df_covid = read_directory_files(all_json)
# df_covid.head()
#
#
# # cleaning text
# def clean_text(text):
#     clean = re.sub(r"[^A-Za-z0-9.,']", " ", text)
#     return clean.lower()
#
# # clean abstract and body_text
# cleaned_abstract = []
# for item in df_covid['abstract']:
#     item = clean_text(item)
#     cleaned_abstract.append(item)
# df_covid['abstract'] = cleaned_abstract
#
# #clean body_text
# cleaned_body = []
# for item in df_covid['body_text']:
#     item = clean_text(item)
#     cleaned_body.append(item)
# df_covid['body_text'] = cleaned_body
#
#
#
# # tokeninzation with nltk - we do not actually use these for modeling as we get embeddings on spacy sentences
# def get_tokens(text):
#     punkt_sentences = sentence_tokenizer.tokenize(text)
#     sentences_words = [treebank_tokenizer.tokenize(sentence) for sentence in punkt_sentences]
#     all_tokens = [word for sentence in sentences_words for word in sentence]
#     all_tokens = [word.lower() for word in all_tokens if word.isalpha()]
#     stop_words = nltk.corpus.stopwords.words('english')
#     all_tokens = [w for w in all_tokens if w not in stop_words]
#     return all_tokens
#
# #tokenize abstract
# abstract_tokens = []
# for item in df_covid['abstract']:
#     item = get_tokens(item)
#     abstract_tokens.append(item)
# df_covid['abstract_tokens'] = abstract_tokens
#
# #tokenize body_text
# body_tokens = []
# for item in df_covid['body_text']:
#     item = get_tokens(item)
#     body_tokens.append(item)
# df_covid['body_tokens'] = body_tokens
#
#
# # save dataframe
# df_covid.to_pickle('./data/preprocessed_dataframe.pkl')
