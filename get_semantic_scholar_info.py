import itertools

# from lib import cached_request
import numpy as np
import pandas as pd
import pickle
import requests
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm


def clean_none(x):
    return x if x is not None else ""


_model = None


def get_model():
    """Singleton for embedding model"""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model


def find_best_author_match(search_results, user):
    # Match abstracts via embedding
    model = get_model()
    reference = model.encode(user["abstracts"])
    reference = reference / np.sqrt((reference**2).sum())
    scores = []

    for r in search_results["data"]:
        the_papers = [
            clean_none(x.get("title")) + " " + clean_none(x.get("abstract"))
            for x in r["papers"][:10]
        ]
        if len(the_papers) == 0:
            scores.append(-1)
            continue
        encodings = model.encode(the_papers, show_progress_bar=True)
        encodings = encodings / np.sqrt((encodings**2).sum(axis=1)).reshape((-1, 1))
        scores.append(reference.dot(encodings.T).max())

    scores = np.array(scores)
    return scores


def main():
    users = pd.read_pickle("data/transformed/match_users_filled.pkl")
    user_matches = {}
    best_scores = []
    inferred_ids = []

    i = 0

    # Find the best matching user in semantic schol by searching for that person's name and comparing to the
    # abstracts they submitted. In theory, people could enter other people's abstracts, but most enter
    # some of their own work, so this works ok.
    for _, user in users.iterrows():
        fullname = (
            user["fullname"]
            .replace("-", " ")
            .replace(".", " ")
            .replace("'", " ")
            .replace(" ", "+")
        )

        q = cached_request(
            f"https://api.semanticscholar.org/graph/v1/author/search?query={fullname}&fields=affiliations,papers.fieldsOfStudy,papers.title,papers.abstract,papers.authors&limit=50"
        )
        scores = find_best_author_match(q, user)

        if len(scores) == 0:
            best_scores.append(0)
            inferred_ids.append(None)
        else:
            best_scores.append(scores.max())
            inferred_ids.append(q["data"][scores.argmax()]["authorId"])

        user_matches[user["user_id"]] = {
            "best_score": best_scores[-1],
            "inferred_id": inferred_ids[-1],
        }
        i += 1

    users["manual_scholar_id"] = users.mindMatchScholarID.str.split("/").map(
        lambda x: x[-1]
        if not isinstance(x, float) and x[-1][0] in "0123456789"
        else np.nan
    )
    users["inferred_scholar_id"] = inferred_ids
    users["inferred_score"] = best_scores

    # We use a threshold match of .4 here, determined by comparing data from
    # manually entered scholar ids and the automatically found ones here.
    # This is a low threshold, since the stakes are pretty low and recall
    # is more important than precision: we want to reject people that
    # already have papers together, and identify people that have a common
    # third coauthor.
    users["consensus_scholar_id"] = np.where(
        users.manual_scholar_id.isna(),
        np.where(users.inferred_score > 0.4, users.inferred_scholar_id, np.nan),
        users.manual_scholar_id,
    )

    # Grab data from all users and figure out the co-authors matrix
    coauthors_map = {}
    for i, user in tqdm(users.iterrows()):
        if user.isna().consensus_scholar_id:
            continue

        q = cached_request(
            f"https://api.semanticscholar.org/graph/v1/author/{user.consensus_scholar_id}?fields=papers.authors"
        )

        min_paper_len = max([min([len(x["authors"]) for x in q["papers"]]), 25])
        coauthors = itertools.chain(
            *[x["authors"] for x in q["papers"] if len(x["authors"]) <= min_paper_len]
        )
        coauthors = {x["authorId"]: x["name"] for x in list(coauthors)}
        del coauthors[q["authorId"]]

        coauthors_map[q["authorId"]] = coauthors

    users.to_pickle("data/transformed/users_w_semantic_scholar.pkl")
    with open("data/transformed/coauthors.pkl", "wb") as f:
        pickle.dump(coauthors_map, f)


if __name__ == "__main__":
    main()
