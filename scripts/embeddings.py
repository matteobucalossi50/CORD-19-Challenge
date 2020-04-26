"""
Use pre-trained model and sciSpacy to extract sentence embeddings with Sentence-Transformers library
"""

# import libraries
import spacy
import pandas as pd
import scispacy
from sentence_transformers import SentenceTransformer

# download pre-trained model
model = SentenceTransformer('bert-large-nli-mean-tokens') #this or the model we trained and saved

# get sent embeddings for each document
def sent_embeddings(df):
    nlp = spacy.load('en_core_sci_md')
    embeddings = []
    for item in df:
        doc = nlp(item)
        sentences = list(doc.sents)
        embeds = model.encode(sentences)
        embeddings.append(embeds)
    return embeddings


# import dataframe
df_covid = pd.read_pickle('./data/preprocessed_dataframe.pkl')  # hopefully this works

# add embeddings to dataframe
df_covid['abs_embeddings'] = sent_embeddings(df_covid['abstract'])
df_covid['body_embeddings'] = sent_embeddings(df_covid['body_text'])

df_covid.head()




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
