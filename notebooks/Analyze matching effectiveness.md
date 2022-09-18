---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: ccn
  language: python
  name: ccn
---

# Analyze mind-matching results

```{code-cell} ipython3
%config InlineBackend.figure_format='retina'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from lib import cached_request, paged_cached_request
from get_semantic_scholar_info import find_best_author_match

dates = {"nmc1": "2020-04-01",
         "nmc2": "2020-05-01",
         "nmc3": "2020-11-01"}

max_coauthors = 25

conference = 'nmc1'

df_users = pd.read_json(f'../data/transformed/{conference}.json', orient='table')
min_date = dates[conference]

infer_identities()

```{code-cell} ipython3
best_scores = []
inferred_ids = []
user_matches = {}
all_scores = []

for _, user in tqdm(df_users.iterrows()):
    fullname = (user['full_name']).replace('-', ' ').replace('.', ' ').replace("'", " ").replace(' ', '+')

    url = f'https://api.semanticscholar.org/graph/v1/author/search?query={fullname}&fields=affiliations,papers.fieldsOfStudy,papers.title,papers.abstract,papers.authors'
    q = paged_cached_request(url)
    scores = np.array(find_best_author_match(q, user.to_dict()['abstracts']))

    if len(scores) == 0:
        best_scores.append(0)
        inferred_ids.append(None)
    else:
        all_scores += list((set(scores) - {scores.max()}))
        best_scores.append(scores.max())
        inferred_ids.append(q['data'][scores.argmax()]['authorId'])

    user_matches[user['registrant_id']] = {'best_score' : best_scores[-1],
                                           'inferred_id': inferred_ids[-1],
                                           'n_matches': len(scores)}
```

Find a minimum threshold to find a match acceptable. Set it so that it's 3 standard deviations higher than the mean max of random people in this dataset. For people whose name only comes up one time, we can use a lower threshold, whereas for people with multiple matching profiles we have to be more stringent.

```{code-cell} ipython3
# Calculate the mean and standard deviation of a max score
all_scores = sorted(all_scores)
all_scores_trunc = np.array(all_scores[int(.02 * len(all_scores)):-int(.02 * len(all_scores))])
#plt.hist(all_scores_trunc)

nboot = 500
threshes = []

for i in range(100):
    idx = np.floor(all_scores_trunc.shape[0] * np.random.rand(nboot, i+1)).astype(int)
    maxes = np.max(all_scores_trunc[idx], axis=1)
    threshes.append(np.mean(maxes) + 3 * np.std(maxes))
```

```{code-cell} ipython3
plt.plot(threshes)
x = np.arange(100)
plt.plot(.515 - (.515 - .33 ) * 2 / (2 + x))
plt.xlabel('Number of matches')
plt.ylabel('3 SD threshold')
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

assert all(df_users['registrant_id'].values == np.array(list(user_matches.keys())))

df_users['ssid'] = [x['inferred_id'] for x in user_matches.values()]
df_users['ssid_score'] = [x['best_score'] for x in user_matches.values()]
df_users['ssid_matches'] = [x['n_matches'] for x in user_matches.values()]

# Use a sliding threshold, where the threshold for those with 1 match is .3 and those with
x = df_users.ssid_matches.values
threshold = .515 - (.515 - .33 ) * 2 / (2 + x)
df_users['ssid'] = np.where(df_users.ssid_score.values > threshold, df_users.ssid, np.nan)
df_users.ssid.isna().mean()
```

```{code-cell} ipython3
df_users[df_users.ssid.isna() & (df_users.abstracts == '')]
```

```{code-cell} ipython3
email = 'hello@xcorr.dev'

def find_open_alex(row):
    name = row.institution.split('(')[0].strip()
    grid_name = row.institution.split('(')[1].split(')')[0]
    query_params = '+'.join(name.split(' '))
    
    # Find the person
    q = requests.get(f"https://api.openalex.org/institutions?search={query_params}&mailto={email}")
    institution_id = None
    for result in q.json()['results']:
        if result['ids']['grid'] == grid_name:
            institution_id = result['id']
            break
            
    if institution_id is None:
        return None

    name = '+'.join(row.full_name.split(' '))
    q = requests.get(f"https://api.openalex.org/authors?search={name}&filter=last_known_institution.id:{institution_id}&mailto={email}")
    return q.json()

#https://api.openalex.org/works?mailto=hello@xcorr.dev
# Find the institution of each person
#q = requests.get(f"https://api.openalex.org/institutions?search=Princeton+University")
#q = requests.get(f"https://api.openalex.org/authors?filter=display_name.search:Qiong+Zhang")
#q.json()

#data = find_open_alex(df_users.loc[194])
#data
```

```{code-cell} ipython3
import requests
apiKey = '6a5600d1a0204c3a88c72563d0cb7166'
url = f'https://api.elsevier.com/content/search/author?query=Qiong+Zhang&apiKey={apiKey}'
q = requests.get(url)
q.json()
```

Not bad! We have found 65% of the people we wanted to. Now let's grab their co-authors list.

```{code-cell} ipython3
df_users.to_json('data/transformed/users.json', orient='table')
```

```{code-cell} ipython3
reidentify_group = []

new_collabs = []
for i, user in tqdm(df_users.iterrows()):
    if user.isna().ssid:
        new_collabs.append(set())
        continue

    q = cached_request(f'https://api.semanticscholar.org/graph/v1/author/{user.ssid}/papers?fields=authors,year,publicationDate')
    
    future_coauthors = []
    past_coauthors = [user.ssid]
    for paper in q['data']:
        if len(paper['authors']) >= max_coauthors:
            continue
    
        if paper['year'] is None:
            continue
        
        if (paper['year'] >= min_year + 1) or (paper['publicationDate'] is not None and paper['publicationDate'] >= min_date):
            # This means we could have influenced this.
            future_coauthors += [x['authorId'] for x in paper['authors']]
        else:
            # Pre-existing coauthorship
            past_coauthors += [x['authorId'] for x in paper['authors']]
            
        reidentify_group += paper['authors']
            
    new_collab = set(future_coauthors) - set(past_coauthors)
    new_collabs.append(new_collab)
```

```{code-cell} ipython3
coauthors_map = {x['name'].lower(): x['authorId'] for x in reidentify_group}
df_ = df_users[df_users.ssid.isna()]
for i, row in df_.iterrows():
    if row['full_name'].strip().lower() in coauthors_map:
        print(row['full_name'])
        print(coauthors_map[row['full_name'].strip().lower()])
        print('Found')
```

```{code-cell} ipython3
from scholarly import scholarly
from scholarly import ProxyGenerator

# Set up a ProxyGenerator object to use free proxies
# This needs to be done only once per session
pg = ProxyGenerator()
pg.FreeProxies()
scholarly.use_proxy(pg)

# Now search Google Scholar from behind a proxy
#search_query = scholarly.search_pubs('Perception of physical stability and center of mass of 3D objects')
#scholarly.pprint(next(search_query))
#scholarly.scholarly.use_proxy
```

```{code-cell} ipython3
import urllib

urllib.parse.parse_qs(
    urllib.parse.urlparse(df_[df_.google_scholar != ''].iloc[4].google_scholar).query
)['user']
```

```{code-cell} ipython3
urllib.parse.urlparse(df_[df_.google_scholar != ''].iloc[0].google_scholar).query
```

```{code-cell} ipython3
author = scholarly.search_author_id('RVO3E
                                    REAAAAJ')
scholarly.fill(author, sections=['basics', 'publications', 'bib'])
author
```

```{code-cell} ipython3
results
```

```{code-cell} ipython3
df_users.google_scholar.iloc[0]
```

```{code-cell} ipython3
df_users[df_users.full_name == 'Pankaj Gupta']
```

```{code-cell} ipython3
df_users['new_collabs'] = new_collabs
```

```{code-cell} ipython3
plt.hist(df_users.new_collabs.map(lambda x: len(x)))
```

```{code-cell} ipython3
(df_users.new_collabs.map(lambda x: len(x)) > 0).mean()
```

53% of people had new collaborators since the Neuromatch conference.

```{code-cell} ipython3
session_ssids = set(df_users[~df_users.ssid.isna()].ssid.values)
```

```{code-cell} ipython3
# Calculate new collabs against a random background
#new_collabs

new_collabs_from_matches = []
new_collabs_from_sessions = []
for i, row in df_users.iterrows():
    user = df_users[df_users.registrant_id == row['registrant_id']]
    if user.empty:
        continue
        
    new_collabs = user.new_collabs.iloc[0]
    match_ssids = set(df_users[df_users.registrant_id.isin([x['registrant_id'] for x in row.matches_info])].ssid.values)
    
    assert len(match_ssids) > 0
    new_collabs_from_match = match_ssids.intersection(new_collabs)
    new_collabs_from_matches.append(len(new_collabs_from_match))
    
    new_collabs_from_session = session_ssids.intersection(new_collabs)
    new_collabs_from_sessions.append(len(new_collabs_from_session))
    
    if new_collabs_from_match:
        print(row)
        print(row['matches_info'])
        print(new_collabs_from_match)
        
print(sum(new_collabs_from_matches) // 2)
print(sum(new_collabs_from_sessions) // 2)
```

```{code-cell} ipython3
df_users[df_users.full_name == 'Nicholas Hardy']
```

```{code-cell} ipython3
df_users[df_users.full_name == 'Jean Laurens'].iloc[0].matches_info
```

```{code-cell} ipython3
df_users[df_users.full_name == 'dun mao'].abstracts
```

```{code-cell} ipython3
df_users[df_users.full_name == 'Jian Liu']
```

```{code-cell} ipython3
# Do a Q & D version of matching
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')

encodings = model.encode(df_users.abstracts.fillna('').values.tolist())
```

```{code-cell} ipython3
match_scores = encodings @ encodings.T
```

Verify that matches are better than chance.

```{code-cell} ipython3
plt.imshow(match_scores)
```

```{code-cell} ipython3
vals = []
vals_ctrl = []
for i, row in df_users.head(df_users.shape[0]).iterrows():
    # Look at the rows and the scores for those rows.
    if row.isna().abstracts:
        continue
    match_indexes = np.array(df_users[~df_users.abstracts.isna() & df_users.registrant_id.isin([x['registrant_id'] for x in row.matches_info])].index.values)
    
    vals += match_scores[i, match_indexes].tolist()
    vals_ctrl += match_scores[(i - 1), match_indexes].tolist()
```

```{code-cell} ipython3
match_indexes
```

```{code-cell} ipython3
plt.hist(vals, np.linspace(0, 1))
```

```{code-cell} ipython3
plt.hist(vals_ctrl, np.linspace(0, 1))
```

```{code-cell} ipython3
#vals_ctrl.mean()
print([np.mean(vals), np.std(vals)])
print([np.mean(vals_ctrl), np.std(vals_ctrl)])
```
