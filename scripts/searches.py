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

    # find closest sentences of corpus for each query sentnece on cosine similarity
    closest_n = 5
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        output = []
        for idx, distance in results[0:closest_n]:
            output.append([corpus[idx].strip(), round(1-distance, 4)])

        table = prettytable.PrettyTable(['Text', 'Score'])
        for i in output:
            text = i[0]
            text = textwrap.fill(text, width=75)
            text = text + '\n\n'
            distance = i[1]
            table.add_row([text, distance])
        print("\n\n======================\n\n")
        print("\nTop 5 most similar sentences in corpus:")
        print(str(table))
        print("\n\n======================\n\n")

# load corpus
df_covid = pd.read_pickle('./data/preprocessed_dataframe.pkl')

# asking the user
query = input('What would you like to know from CORD-19? ')
print('\nUse abstracts:')
sem_search(query, embedder, df_covid['abstract'], df_covid['abs_embeddings'])

print('\nUse full text:')
sem_search(query, embedder, df_covid['body_text'], df_covid['body_embeddings'])

