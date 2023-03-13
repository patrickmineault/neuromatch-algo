import itertools
from pyalex import Works

import pandas as pd
import pickle
from tqdm import tqdm


def main():
    users = pd.read_pickle("data/transformed/match_users_openalex.pkl")

    # Grab data from all users and figure out the co-authors matrix
    coauthors_map = {}
    for i, user in tqdm(users.iterrows()):
        if user.isna().inferred_id:
            continue

        works = (
            Works()
            .filter(author={"id": user.inferred_id})
            .sort(publication_date="desc")
            .get()
        )

        min_paper_len = max([min([len(x["authorships"]) for x in works]), 25])
        coauthors = itertools.chain(
            *[x["authorships"] for x in works if len(x["authorships"]) <= min_paper_len]
        )
        coauthors = {
            x["author"]["id"]: x["author"]["display_name"] for x in list(coauthors)
        }
        del coauthors[user.inferred_id]

        coauthors_map[user.inferred_id] = coauthors

    with open("data/transformed/coauthors.pkl", "wb") as f:
        pickle.dump(coauthors_map, f)


if __name__ == "__main__":
    main()
