import collections
import itertools

from scipy.cluster.hierarchy import linkage
import hcluster   # requires dedupe-hcluster
from paper_reviewer_matcher import (
    preprocess, compute_affinity
)

from cmath import isnan
import numpy as np
import pandas as pd
from paper_reviewer_matcher.group_matching import compute_conflicts, generate_pod_numbers
import pickle
from tqdm import tqdm

from scipy.spatial.distance import squareform

PPL_PER_GROUP = 4
NROUNDS = 3


def measure_goodness(A_cluster, cluster_assignments):
    dists = []
    for i in range(cluster_assignments.min(), cluster_assignments.max()+1):
        # Calculate the average pairwise distance within the cluster.
        d = A_cluster[cluster_assignments == i, :][:, cluster_assignments == i]
        if (d == 1).any():
            mean_dist = 1
        else:
            mean_dist = d.mean()
        dists.append(mean_dist)
        
    return dists

def get_distance_vector(B):
    Bs = (B + B.T) / 2.0
    diag_mask = (np.ones_like(Bs) - np.eye(Bs.shape[0]))
    Bs = Bs * diag_mask
    return squareform(Bs)

def agglomerate(A, group_size):
    ngroups = int(np.ceil(A.shape[0] / group_size))
    nsmallgroups = ngroups * group_size - A.shape[0]
    nbiggroups = ngroups - nsmallgroups
    labels = np.ones(A.shape[0]) * np.nan
    
    A = A.copy()
    
    groups = []
    
    if group_size == 2:
        # Round up with group sizes of 2
        if nsmallgroups == 1:
            group_sizes = [group_size + 1] + [group_size] * (nbiggroups - 1)
        else:
            group_sizes = [group_size] * nbiggroups
    else:
        group_sizes = [group_size] * nbiggroups + [group_size - 1] * nsmallgroups
    assert A.shape[0] == sum(group_sizes)
    j = 0
    for gs in tqdm(group_sizes):
        B = A[np.isnan(labels), :][:, np.isnan(labels)]
        z = linkage(get_distance_vector(B),
                    method='average',
                    metric='euclidean')
        
        the_nums = np.where(z[:, -1] >= gs)[0]
        minpos = the_nums.min()
        
        cluster_nums = [z[minpos, 0], z[minpos, 1]]
        
        i = 0
        while i < len(cluster_nums):
            if cluster_nums[i] >= B.shape[0]:
                cluster_nums.append(z[int(cluster_nums[i]) - B.shape[0], 0])
                cluster_nums.append(z[int(cluster_nums[i]) - B.shape[0], 1])
            i += 1
            
        cluster_nums = np.array(cluster_nums).astype(int)
        cluster_nums = cluster_nums[cluster_nums < B.shape[0]]
        
        assert len(cluster_nums) >= gs
        cluster_nums = cluster_nums[:gs]
        
        # Map cluster nums to the original numbers prior to subsetting.
        the_map = np.where(np.isnan(labels))[0]        
        cluster_nums = [the_map[k] for k in cluster_nums]
        labels[cluster_nums] = j        
        j += 1
        
    return labels.astype(int)


def do_matching(M, ppl_per_group, nrounds, fake=False):
    # Ban previous match sets
    A = M.copy()

    if fake:
        A = .01 * np.random.randn(A.shape[0], A.shape[1])
        A[M == 1] = 1

    std_per = 0.02

    labels = agglomerate(A + np.random.randn(A.shape[0]) * std_per, ppl_per_group)
    goodnesses = np.array(measure_goodness(M, labels))
    print([goodnesses.mean(), np.std(goodnesses)])

    groups = [labels]
    all_goodnesses = [goodnesses]

    for j in range(nrounds - 1):
        for i in range(labels.max()+1):
            a = np.where(labels==i)[0]
            for k in a:
                A[labels==i, k] = 1
                
        print((A == 1).sum())

        labels = agglomerate(A + np.random.randn(A.shape[0]) * std_per, PPL_PER_GROUP)

        goodnesses = np.array(measure_goodness(M, labels))
        print([goodnesses.mean(), np.std(goodnesses)])

        groups.append(labels)
        all_goodnesses.append(goodnesses)

    return groups, all_goodnesses


def calculate_common_coauthors(users, matches, coauthors_map):
    all_authors = collections.ChainMap(*list(coauthors_map.values()))
    common_coauthors = []
    coauthor_pairs = []
    for _, group in matches.iterrows():
        scholar_ids = users[users.user_id.isin(group['user_ids'])].consensus_scholar_id.tolist()
        scholar_ids = [x for x in scholar_ids if isinstance(x, str)]

        # Also count common co-authors.
        common_pair = 0
        for s in scholar_ids:
            for k in coauthors_map.get(s).keys():
                if k in scholar_ids:
                    common_pair += 1/2

        counts = collections.Counter(
            list(itertools.chain(*[list(coauthors_map.get(x).keys()) for x in scholar_ids]))
        )
        in_common = []
        for x, n in counts.most_common():
            if n >= 2:
                in_common.append(all_authors[x])
        common_coauthors.append(in_common)
        coauthor_pairs.append(common_pair)

    return coauthor_pairs, common_coauthors

def main():
    users = pd.read_pickle('data/transformed/users_w_semantic_scholar.pkl')
    M = np.load('data/transformed/match_matrix.npy')
    with open('data/transformed/coauthors.pkl', 'rb') as f:
        coauthors_list = pickle.load(f)

    cois_list = compute_conflicts(users)
    for i, j in cois_list:
        M[i, j] = 1

    print(f"Banned matches {(M==1).sum() / 2}")

    # Although we didn't specifically say to people that they won't be matched with their coauthors,
    # I think it's only right.
    # coauthors_mat = np.zeros_like(M)
    scholar_ids = users['consensus_scholar_id'].tolist()
    for i, row in users.iterrows():
        scholar_id = row['consensus_scholar_id']
        if isinstance(scholar_id, float) and np.isnan(scholar_id):
            continue

        coauthors = coauthors_list[scholar_id]
        for c in coauthors.keys():
            if c in scholar_ids:
                # Find it and ban
                j = users[users.consensus_scholar_id == c].index[0]
                M[i, j] = 1

    print(f"Banned matches following co-author banning {(M==1).sum() / 2}")

    # Do random matching by overwriting the affinity matrix
    groups, goodness = do_matching(M, PPL_PER_GROUP, NROUNDS, fake=True)
    matches = []
    for j, round in enumerate(groups):
        for i in range(int(max(round)+1)):
            matches.append({'user_ids': users.iloc[np.where(round == i)].user_id.tolist(),
                            'round': j,
                            'group': i,
                            'goodness': goodness[j][i]})

    df_matches = pd.DataFrame(matches)
    df_matches.to_pickle('data/output/random_matches.pkl')

    # Now do three rounds of matching.
    groups, goodness = do_matching(M, PPL_PER_GROUP, NROUNDS)

    # Assemble all the groups.
    matches = []
    for j, round in enumerate(groups):
        for i in range(int(max(round)+1)):
            matches.append({'user_ids': users.iloc[np.where(round == i)].user_id.tolist(),
                            'round': j,
                            'group': i,
                            'goodness': goodness[j][i]})

    df_matches = pd.DataFrame(matches)

    # Measure common co-authors.
    direct_coauthors, indirect_coauthors = calculate_common_coauthors(users, df_matches, coauthors_list)

    assert sum(direct_coauthors) == 0  # By definition

    # Count the indirect coauthors
    df_matches['indirect_coauthors'] = indirect_coauthors
    df_matches['has_indirect_coauthors'] = df_matches['indirect_coauthors'].map(lambda x: len(x) > 0)

    print(df_matches.groupby('round').has_indirect_coauthors.sum())
    print(df_matches.groupby('round').goodness.mean())

    df_matches.to_pickle('data/output/matches.pkl')
    df_matches.to_json('data/output/matches.json')

if __name__ == '__main__':
    main()