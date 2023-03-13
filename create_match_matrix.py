NROUNDS = 3
PPL_PER_GROUP = 4

from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
import time


from scipy.cluster.hierarchy import linkage
import hcluster  # requires dedupe-hcluster
from paper_reviewer_matcher import preprocess, compute_affinity

import dbm
import json

from paper_reviewer_matcher.group_matching import (
    compute_conflicts,
    generate_pod_numbers,
)


def clean_up(x):
    return x.replace("\n", " ").replace("  ", " ")


def deconcatenate_abstracts(abstracts):
    # Split concatenated abstracts into individual abstracts
    abstract_index = []
    all_abstracts = []
    for i, a in enumerate(abstracts):
        b = [
            x.strip()
            for x in re.split(r"(\n\n|[\s\.]\[)", a.strip())
            if x.strip() != "" and x.strip() != "["
        ]

        # Clean up to remove small titles clogging up the list of separate abstracts
        mean_len = sum([len(x) for x in b]) / len(b)
        buff = ""
        b_out = []
        for ab in b:
            if len(ab) < mean_len / 2:
                # Keep it in the buffer
                buff += ab + " "
            else:
                b_out.append((buff + " " + ab).strip())
                buff = ""

        if buff != "":
            b_out[-1] = b_out[-1] + buff

        b = b_out
        all_abstracts += b
        abstract_index += [i] * len(b)
    return abstract_index, all_abstracts


def chop(sentences, start, end):
    return [" ".join(x.split(" ")[start:end]) for x in sentences]


def embed(abstracts):
    # Embed 250 words at a time and pool the resulting embeddings post-hoc
    piece_size = 250
    model = SentenceTransformer("all-mpnet-base-v2")
    embeds = np.zeros((len(abstracts), 768))
    abstracts = [x.strip() for x in abstracts]
    for i in range(5):
        chopped = chop(abstracts, piece_size * i, piece_size * (i + 1))
        abstracts_embedded = model.encode(chopped, show_progress_bar=True)
        embeds += abstracts_embedded * np.array([len(x) for x in chopped]).reshape(
            (-1, 1)
        )

    embeds = embeds / np.sqrt((embeds**2).sum(axis=1, keepdims=True))
    return embeds


def main():
    users = pd.read_pickle("data/transformed/match_users_openalex.pkl")

    users.Abstract1 = np.where(
        ~users.Abstract1.isna(), users.Abstract1, users.inferred_abstract_1
    )
    users.Abstract2 = np.where(
        ~users.Abstract2.isna(), users.Abstract2, users.inferred_abstract_2
    )
    users.Abstract3 = np.where(
        ~users.Abstract3.isna(), users.Abstract3, users.inferred_abstract_3
    )

    jail_tokens = "I didn't do the requested task, therefore I get renaissance paintings with green eggs and ham instead of matches"

    users.Abstract1 = np.where(
        ~users.Abstract1.isna(), users.Abstract1, [jail_tokens] * len(users)
    )

    # Go from concatenated abstracts to individual ones.
    abstract_index = (
        list(range(len(users))) + list(range(len(users))) + list(range(len(users)))
    )

    all_abstracts = (
        users.Abstract1.tolist() + users.Abstract2.tolist() + users.Abstract3.tolist()
    )

    abstract_index, all_abstracts = zip(
        *[
            (i, a)
            for i, a in zip(abstract_index, all_abstracts)
            if not isinstance(a, float) and a is not None
        ]
    )

    # Encode each abstract separately.
    individual_encodings = embed(all_abstracts)

    D = individual_encodings @ individual_encodings.T
    D = D.ravel()

    # We use a rule that a person's match to another is determined by their best matched abstracts,
    # rather than their average. Aggregate across all the abstracts shared between two people to
    # find the max.
    first_index = np.array(abstract_index).reshape((-1, 1)) @ np.ones(
        (1, len(abstract_index))
    )
    second_index = first_index.T

    first_index = first_index.ravel()
    second_index = second_index.ravel()

    first_abstract = np.arange(len(abstract_index)).reshape((-1, 1)) @ np.ones(
        (1, len(abstract_index))
    )
    second_abstract = first_abstract.T

    first_abstract = first_abstract.ravel()
    second_abstract = second_abstract.ravel()

    df_individual_match = pd.DataFrame(
        {
            "first_participant": pd.Series(first_index),
            "second_participant": pd.Series(second_index),
            "first_abstract": pd.Series(first_abstract),
            "second_abstract": pd.Series(second_abstract),
            "match": pd.Series(D.ravel()),
        }
    )

    df_idxmax = df_individual_match.groupby(
        ["first_participant", "second_participant"]
    ).idxmax()
    nums = df_idxmax.match

    # translate nums into left abstract, top abstract
    first_abstract, second_abstract = nums % len(all_abstracts), nums // len(
        all_abstracts
    )

    df_idxmax["first_abstract"] = first_abstract
    df_idxmax["second_abstract"] = second_abstract

    df_idxmax.to_pickle("data/transformed/abstract_indices.pkl")

    with open("data/transformed/all_abstracts.pkl", "wb") as f:
        pickle.dump(all_abstracts, f)

    the_best = df_individual_match.groupby(
        ["first_participant", "second_participant"]
    ).max()

    # The result: a match matrix.
    M = the_best.match.values.reshape((len(users), len(users)))
    M = (1 - M) / 2.0

    np.save("data/transformed/match_matrix.npy", M)


if __name__ == "__main__":
    main()
