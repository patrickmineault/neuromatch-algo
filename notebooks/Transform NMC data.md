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

```{code-cell} ipython3
# Load up data
import json
import pandas as pd

def rename_rows(rows):
    for row in rows:
        row['full_name'] = row['fullname']
        row['registrant_id'] = row['id']
    return rows

for edition in [1, 2, 3]:

    with open(f'../data/raw/users_2020_{edition}.json', 'r') as f:
        user_data = json.load(f)
        
    with open(f'../data/raw/matches_2020_{edition}.json', 'r') as f:
        match_data = json.load(f)

    for k, user in user_data.items():
        user['registrant_id'] = k
        
    df_users = pd.DataFrame(user_data.values())
    df_users = df_users.rename(columns={'fullname': 'full_name'})
    df_users.abstracts = df_users.abstracts.map(lambda x: '\n\n'.join(x))

    matches = []
    for k, match in match_data.items():
        matches.append(
            {'registrant_id': k,
             'matches_info': rename_rows(match['group_match'] + match['mind_match'])}
        )

    df_users = df_users.merge(pd.DataFrame(matches), left_on='registrant_id', right_on='registrant_id')
    df_users = df_users.sort_values('registrant_id').reset_index(drop=True)
    df_users.to_json(f'../data/processed/nmc{edition}.json', orient='table')
```
