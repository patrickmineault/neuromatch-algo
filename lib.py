import Levenshtein as levenshtein
from joblib import Memory
import json
import numpy as np
import requests
import scholarly
from scholarly import scholarly
from scholarly import ProxyGenerator, _proxy_generator
import time
import urllib
from urllib.parse import urlencode

from get_semantic_scholar_info import find_best_author_match

pg = ProxyGenerator()
pg.FreeProxies()
scholarly.use_proxy(pg)

memory = Memory("cachedir", verbose=0)


@memory.cache
def cached_request(url):
    wait_time = 5
    waited = 0
    q = None
    while waited <= 3:
        q = requests.get(url)
        if q.status_code == 200:
            break

        # Exponential retry
        time.sleep(wait_time)
        wait_time *= 2
        waited += 1
    if q is None:
        return None
    else:
        return q.json()


def paged_cached_request(url):
    page_size = 10
    max_results = 100
    results = cached_request(f"{url}&offset=0&limit={page_size}")
    all_results = results
    page = 0
    while "next" in results:
        page += 1
        results = cached_request(f"{url}&offset={results['next']}&limit={page_size}")
        if results is None or "data" not in results:
            results = {"next": page * page_size}
            continue

        all_results["data"] += results["data"]
        if len(all_results) >= max_results:
            break

    return all_results


@memory.cache
def infer_author_id_from_google(url):
    """Find an author by ID on Google Scholar, find their publications,
    search for their publications on semantic scholar, find their ID on semantic scholar that way"""
    parsed_query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if "user" not in parsed_query:
        return None

    # Fetch info
    try:
        author = scholarly.search_author_id(parsed_query["user"][0])

        data = scholarly.fill(author, sections=["basics", "publications"])

        author_name = data["name"]

        titles = []
        author_pos = []
        candidate_author_ids = set()
        for pub in data["publications"]:
            filled_pub = scholarly.fill(pub)
            author_list = filled_pub["bib"]["author"].split(" and ")

            # Find the best matching out of the list
            distances = [
                (levenshtein.distance(x, author_name), i)
                for i, x in enumerate(author_list)
            ]
            _, best_idx = min(distances)

            titles.append(filled_pub["bib"]["title"])
            author_pos.append(best_idx)

            # Now find the equivalent publication in semantic scholar
            title = urllib.parse.quote_plus(filled_pub["bib"]["title"])

            q = cached_request(
                f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&fields=authors.authorId,authors.name&limit=1"
            )
            if len(q["data"]) == 0:
                continue
            authors_ss = q["data"][0]["authors"]

            distances = [
                (levenshtein.distance(x["name"], author_name), i)
                for i, x in enumerate(authors_ss)
            ]
            _, best_idx_ss = min(distances)

            if best_idx_ss == best_idx:
                author_id_ss = authors_ss[best_idx_ss]["authorId"]
                if author_id_ss in candidate_author_ids:
                    # Two votes – bingo!
                    return author_id_ss

                candidate_author_ids = candidate_author_ids.union(set([author_id_ss]))
    except _proxy_generator.MaxTriesExceededException:
        return None

    return None


def thresh(nmatches):
    """Determined by simulation in some early data, where it corresponded to 3 sigmas away from chance."""
    return 0.515 - (0.515 - 0.33) * 2 / (2 + nmatches)


def infer_author_id(data):
    """Infer semantic scholar id from data about an author."""

    # Use Google Scholar if available
    if "google_scholar" in data and data["google_scholar"] != "":
        author_id = infer_author_id_from_google(data["google_scholar"])
        if author_id is not None:
            return author_id

    # Otherwise, use semantic scholar API to find the best matching abstract.
    fullname = urllib.parse.quote_plus(
        data["full_name"].replace("-", " ").replace(".", " ").replace("'", " ")
    )

    url = f"https://api.semanticscholar.org/graph/v1/author/search?query={fullname}&fields=affiliations,papers.fieldsOfStudy,papers.title,papers.abstract,papers.authors"
    q = paged_cached_request(url)

    scores = np.array(find_best_author_match(q, data.to_dict()))

    if len(scores) > 0 and scores.max() > thresh(len(scores)):
        return q["data"][scores.argmax()]["authorId"]

    return None


if __name__ == "__main__":
    # data = infer_author_id_from_google(
    #    "https://scholar.google.com/citations?user=gpQg9uQAAAAJ&hl=en"
    # )
    # print(data)
    paged_cached_request(
        "https://api.semanticscholar.org/graph/v1/author/search?query=Patrick+Mineault&fields=affiliations,papers.fieldsOfStudy,papers.title,papers.abstract,papers.authors&limit=50"
    )
