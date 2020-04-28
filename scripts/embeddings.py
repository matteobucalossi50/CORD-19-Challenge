"""
Use pre-trained model and sciSpacy to extract sentence embeddings with Sentence-Transformers library
"""

# import libraries
import spacy
import pandas as pd
import scispacy
from sentence_transformers import SentenceTransformer

# download pre-trained model
# model = SentenceTransformer('training_nli_allenai-scibert_scivocab_uncased-2020-04-26_13-22-06') #this or the model we trained and saved

# get sent embeddings for each document
def sent_embeddings_wr(corpus, model):
    # nlp = spacy.load('en_core_sci_md') # not sure if needed cause not sure we have to embed by sents, prolly just by doc
    embeddings = []
    for item in corpus:
        # doc = nlp(item)
        # sentences = list(doc.sents)
        embeds = model.encode(item)
        embeddings.append(embeds)
    return embeddings


# embeddings
def sent_embeddings(corpus, model):
    embeddings = model.encode(corpus)
    return embeddings


# import dataframe
# df_covid = pd.read_pickle('/Users/Matteo/Desktop/ML1/project/data/preprocessed_dataframe.pkl')  # hopefully this works

# convert to list for transformer
# def list_conv(df):
#     corpus_list = []
#     for item in df:
#         corpus_list.append(item)
#     return corpus_list
#
# df_covid['abstract'] = (df_covid['abstract']).tolist()
# df_covid['body_text'] = (df_covid['body_text']).tolist()

# add embeddings to dataframe
# df_covid['abs_embeddings'] = sent_embeddings(df_covid['abstract'], model)
# df_covid['body_embeddings'] = sent_embeddings(df_covid['body_text'], model)

# save dataframe
# df_covid.to_pickle('./data/preprocessed_dataframe.pkl')

# df_covid.head()




###
# abs_embeddings = []
# for item in df_covid['abstract']:
#     doc = nlp(item)
#     sentences = list(doc.sents)
#     embeddings = model.encode(sentences)
#     abs_embeddings.append(embeddings)
# df_covid['abs_embeddings'] = abs_embeddings
# df_covid.head()
#
#
# body_embeddings = []
# for item in df_covid['body_text']:
#     doc = nlp(item)
#     sentences = list(doc.sents)
#     embeddings = model.encode(sentences)
#     body_embeddings.append(embeddings)
# df_covid['body_embeddings'] = body_embeddings
# df_covid.head()
###
