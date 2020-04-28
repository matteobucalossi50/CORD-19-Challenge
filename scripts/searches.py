import os
import prettytable
import scipy.spatial
import pandas as pd
import pickle
import tqdm
import textwrap
from sentence_transformers import SentenceTransformer
import warnings
warnings.simplefilter('ignore')

# import model
embedder = SentenceTransformer('bert-large-nli-mean-tokens') #this or the model we trained and saved


def sem_search(query, model, corpus, corpus_embeddings):
    queries = [query]
    query_embeddings = model.encode(queries)

    # find closest sentences of corpus for each query sentence on cosine similarity
    closest_n = 5
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        output = []
        for idx, distance in results[0:closest_n]:
            output.append([corpus.iloc[idx, 1].strip(), corpus.iloc[idx, 3].strip(),
                           corpus.iloc[idx, 4].strip(), corpus.iloc[idx, 5].strip(), round(1-distance, 4)])


        table = prettytable.PrettyTable(['Abstract','Authors','Title','Journal','Score'])
        for i in output:
            abstract = i[0]
            abstract = textwrap.fill(abstract, width=75)
            abstract = abstract + '\n\n'
            author = i[2]
            author = textwrap.fill(author, width=75)
            author = author + '\n\n'
            title = i[3]
            title = textwrap.fill(title, width=75)
            title = title + '\n\n'
            journal = i[4]
            journal = textwrap.fill(journal, width=75)
            journal = journal + '\n\n'
            distance = i[5]
            table.add_row([abstract, author, title, journal, distance])
        print("\n\n======================\n\n")
        print("\nTop 5 most similar articles in corpus:")
        print(str(table))
        print("\n\n======================\n\n")


# load corpus
df_covid = pd.read_pickle('preprocessed_dataframe_withabs.pkl')

# asking the user
query = input('What would you like to know from CORD-19? ')
print('\nUse abstracts:')
print(sem_search(query, embedder, df_covid, df_covid['abs_embeddings']))

print('\nUse full text:')
sem_search(query, embedder, df_covid, df_covid['body_embeddings'])

