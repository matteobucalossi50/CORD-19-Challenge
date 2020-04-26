import os
import prettytable
import scipy.spatial
import pandas as pd
import pickle
import tqdm
import textwrap
import warnings
warnings.simplefilter('ignore')

# load corpus embeddings

# queries

queries = [query]
#query_embeddings =

# find closest sentences of corpus for each query sentnece on cosine similarity
closest_n = 5
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))


# make them function and output a prettytable.PrettyTable


