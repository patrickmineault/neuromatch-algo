from lib import get_model, find_best_author_match, cached_request
import numpy as np
import pandas as pd
import tqdm

import pyalex
from pyalex import Authors, Works, Institutions

pyalex.config.email = "patrick.mineault@gmail.com"


def resolve_from_orcid(orcid):
    author = Authors().filter(orcid=orcid).get()
    if author:
        return author[0]["id"]
    else:
        return None


def main():
    # For each user, find their OpenAlex ID as well as the OpenAlex ID of their institution.
    # * If the person has written down an ORCID, we will use that to match them to the database
    # * If not, we will search for them. We will use the same criteria as we used with semantic scholar.
    # * If not, we will compare them with the average embedding for the conference. If they are above .25, we will use that ID.
    users = pd.read_pickle("data/transformed/match_users_filled.pkl")

    matches_by_interest = {}

    # Calculate the average interest of CoSyNe users.
    model = get_model()
    average_interest = model.encode(
        np.concatenate(
            [
                users[~users.isna().Abstract1].Abstract1.values,
                users[~users.isna().Abstract2].Abstract2.values,
                users[~users.isna().Abstract3].Abstract3.values,
            ]
        )
    ).mean(axis=0)

    i = 0

    # Find the best matching user in semantic schol by searching for that person's name and comparing to the
    # abstracts they submitted. In theory, people could enter other people's abstracts, but most enter
    # some of their own work, so this works ok.
    inferred_id = []
    inferred_score = []
    inferred_method = []
    inferred_abstract_1 = []
    inferred_abstract_2 = []
    inferred_abstract_3 = []

    score_types = {True: "average_interest", False: "abstract_match"}

    for _, user in tqdm.tqdm(users.iterrows()):
        if not user.isna().ORCID:
            # Try that first
            the_id = resolve_from_orcid(user.ORCID)
            if the_id is not None:
                inferred_id.append(the_id)
                inferred_score.append(1)
                inferred_method.append("ORCID")
                inferred_abstract_1.append(None)
                inferred_abstract_2.append(None)
                inferred_abstract_3.append(None)
                continue

        # If that didn't work, ask OpenAlex
        records = []
        hits = Authors().search(user["fullname"]).get()

        for hit in hits:
            the_id = hit["id"]
            works = (
                Works()
                .filter(author={"id": the_id})
                .sort(publication_date="desc")
                .get()
            )

            the_titles = [work["title"] for work in works]
            the_abstracts = [work["abstract"] for work in works]
            abstract_set = set()
            valid_idx = []
            for i, abstract in enumerate(the_abstracts):
                if (
                    abstract is not None
                    and abstract != ""
                    and (abstract not in abstract_set)
                ):
                    valid_idx.append(i)
                    abstract_set.add(abstract)
                if len(valid_idx) >= 10:
                    break

            records.append(
                {
                    "author": hit,
                    "papers": [
                        {"title": the_titles[i], "abstract": the_abstracts[i]}
                        for i in valid_idx
                    ],
                }
            )

        q = {"data": records}
        scores, score_type = find_best_author_match(q, user, average_interest, False)

        if len(scores) == 0:
            inferred_score.append(0)
            inferred_id.append(None)
            inferred_abstract_1.append(None)
            inferred_abstract_2.append(None)
            inferred_abstract_3.append(None)
        else:
            inferred_score.append(scores.max())
            inferred_id.append(q["data"][scores.argmax()]["author"]["id"])

            abstracts = [x["abstract"] for x in q["data"][scores.argmax()]["papers"]]

            if score_type == True:
                inferred_abstract_1.append(abstracts[0] if len(abstracts) > 0 else None)
                inferred_abstract_2.append(abstracts[1] if len(abstracts) > 1 else None)
                inferred_abstract_3.append(abstracts[2] if len(abstracts) > 2 else None)
            else:
                inferred_abstract_1.append(None)
                inferred_abstract_2.append(None)
                inferred_abstract_3.append(None)

        inferred_method.append(score_types[score_type])

    # Also find the academic affiliation of the user.
    institutions = []
    for _, user in tqdm.tqdm(users.iterrows()):
        institution = Institutions().search(user["Company"]).get()
        if institution:
            institution = institution[0]["id"]
            institutions.append(institution)
        else:
            # Just search for the first word in the company name.
            institution = Institutions().search(user["Company"].split(" ")[0]).get()
            if institution and institution[0]["relevance_score"] > 100_000:
                institution = institution[0]["id"]
                institutions.append(institution)
            else:
                institutions.append(None)

    users["institution"] = institutions

    users["inferred_id"] = inferred_id
    users["inferred_score"] = inferred_score
    users["inferred_method"] = inferred_method
    users["inferred_abstract_1"] = inferred_abstract_1
    users["inferred_abstract_2"] = inferred_abstract_2
    users["inferred_abstract_3"] = inferred_abstract_3

    users["inferred_id"] = np.where(
        (users["inferred_score"] < 0.25)
        | (
            (users["inferred_method"] != "average_interest")
            & (users["inferred_score"] < 0.4)
        ),
        np.nan,
        users["inferred_id"],
    )

    users.to_pickle("data/transformed/match_users_openalex.pkl")


if __name__ == "__main__":
    main()
