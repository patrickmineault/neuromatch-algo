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
import hcluster   # requires dedupe-hcluster
from paper_reviewer_matcher import (
    preprocess, compute_affinity
)

import dbm
import json

from paper_reviewer_matcher.group_matching import compute_conflicts, generate_pod_numbers

def clean_up(x):
    return x.replace('\n', ' ').replace('  ', ' ')

def deconcatenate_abstracts(abstracts):
    # Split concatenated abstracts into individual abstracts
    abstract_index = []
    all_abstracts = []
    for i, a in enumerate(abstracts):
        b = [x.strip() for x in re.split(r"(\n\n|[\s\.]\[)", a.strip()) if x.strip() != '' and x.strip() != '[']

        # Clean up to remove small titles clogging up the list of separate abstracts
        mean_len = sum([len(x) for x in b]) / len(b)
        buff = ''
        b_out = []
        for ab in b:
            if len(ab) < mean_len / 2:
                # Keep it in the buffer
                buff += ab + ' '
            else:
                b_out.append((buff + ' ' + ab).strip())
                buff = ''

        if buff != '':
            b_out[-1] = b_out[-1] + buff

        b = b_out
        all_abstracts += b
        abstract_index += [i] * len(b)
    return abstract_index, all_abstracts

def main():
    users = pd.read_csv('data/input/CCN2022MindMatchList_2022-Aug-15-000241.csv')

    # Fill in abstract for the people who didn't fill them
    # Found from this page: http://www.theses.fr/s339475
    # https://grantome.com/grant/NSF/IIS-1912280
    users.loc[users.RegistrantID == 1110, 'RepresentativeWork'] = 'Oscillatory processes in deep neural networks\nThe development of deep convolutional networks (DCNs) has recently led to great successes in machine vision. Despite these successes, to date, the most impressive results have been obtained for image categorization tasks such as indicating whether an image contains a particular object. However, DCNs ability to solve more complex visual reasoning problems such as understanding the visual relations between objects remains limited. Interestingly, much work in computer vision is currently being devoted to extending DCNs, but these models are still outmatched by the power and versatility of the brain, perhaps in part due to the richer neuronal computations available to cortical circuits. The challenge is to identify which neuronal mechanisms are relevant and to find suitable abstractions to model them. One promising set of candidates is the neural oscillations that are found throughout the brain. This project seeks to identify the key oscillatory components and characterize the neural computations underlying humans ability to solve visual reasoning tasks, and to use similar strategies in modern deep learning architectures.This project will use existing computational models to develop tasks and stimuli to be used in EEG studies to identify the key oscillatory components underlying human visual reasoning ability. The analysis of these EEG data will be guided by the development of a biophysically-realistic computational neuroscience model. This will inform the development of hypotheses on the circuit mechanisms underlying the oscillatory clusters and relate these mechanisms to neural computations. Finally, the project will develop novel machine learning idealizations of these neural computations, which are trainable with current deep learning methods but still interpretable at the neural circuit level. In particular, the project will further develop initial machine learning formulation of oscillations based on complex-valued neuronal units, thus extending the approach and demonstrating its ability to qualitatively capture key oscillatory processes underlying visual reasoning.'

    # This person I can't find online, so am filling it with the top paper of each person they'd like to meet
    users.loc[users.RegistrantID == 1228, 'RepresentativeWork'] = "We propose an algorithm for meta-learning that is model-agnostic, in the sense that it is compatible with any model trained with gradient descent and applicable to a variety of different learning problems, including classification, regression, and reinforcement learning. The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples. In our approach, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task. In effect, our method trains the model to be easy to fine-tune. We demonstrate that this approach leads to state-of-the-art performance on two few-shot image classification benchmarks, produces good results on few-shot regression, and accelerates fine-tuning for policy gradient reinforcement learning with neural network policies.\n\nComputational modeling and behavioral experimentation suggest that human frontal lobe function is capable of monitoring three or four concurrent behavioral strategies in order to select the most suitable one during decision-making.\n\nTheories of predictive processing propose that prediction error responses are modulated by the certainty of the predictive model or precision. While there is some evidence for this phenomenon in the visual and, to a lesser extent, the auditory modality, little is known about whether it operates in the complex auditory contexts of daily life. Here, we examined how prediction error responses behave in a more complex and ecologically valid auditory context than those typically studied. We created musical tone sequences with different degrees of pitch uncertainty to manipulate the precision of participants' auditory expectations. Magnetoencephalography was used to measure the magnetic counterpart of the mismatch negativity (MMNm) as a neural marker of prediction error in a multi-feature paradigm. "

    # I got this from the top 3 papers here https://scholar.google.com/citations?hl=en&user=pSE26qUAAAAJ&view_op=list_works
    users.loc[users.RegistrantID == 1424, "RepresentativeWork"] = "Due to the nature of fMRI acquisition protocols, slices in the plane of acquisition are not acquired simultaneously or sequentially, and therefore are temporally misaligned with each other. Slice timing correction (STC) is a critical preprocessing step that corrects for this temporal misalignment. Interpolation-based STC is implemented in all major fMRI processing software packages. To date, little effort has gone towards assessing the optimal method of STC. Delineating the benefits of STC can be challenging because of its slice-dependent gain as well as its interaction with other fMRI artifacts. In this study, we propose a new optimal method (Filter-Shift) based on the fundamental properties of sampling theory in digital signal processing. We then evaluate our method by comparing it to two other methods of STC from the most popular statistical software packages, SPM and FSL.\n\nUse of functional magnetic resonance imaging (fMRI) in studies of aging is often hampered by uncertainty about age‐related differences in the amplitude and timing of the blood oxygenation level dependent (BOLD) response (i.e., hemodynamic impulse response function (HRF)). Such uncertainty introduces a significant challenge in the interpretation of the fMRI results. Even though this issue has been extensively investigated in the field of neuroimaging, there is currently no consensus about the existence and potential sources of age‐related hemodynamic alterations. Using an event‐related fMRI experiment with two robust and well‐studied stimuli (visual and auditory), we detected a significant age‐related difference in the amplitude of response to auditory stimulus. \n\nIn this paper, we use convolutional neural networks (CNNs) to model/capture the relationship between simultaneously acquired EEG and fMRI. Specifically we use CNNs to implement neural transcoding - i.e. generating one neuroimaging modality from another - from EEG to fMRI and vice versa. The novelty of our approach lies in its ability to resolve the source space without prior hemodynamic and leadfield estimation. The two CNNs, one for EEG-to-fMRI and the other fMRI-to-EEG transcoding, are coupled in their source space representations, and given their architecture are able to capture both linear and non-linear transformations that map two imaging modalities into a common neural source space. We present results on simulated simultaneously acquired EEG-fMRI data and show the performance of mapping each modality to the other."

    users['abstracts'] = users.RepresentativeWork.map(clean_up)
    users = users.rename(columns={'RegistrantID': 'user_id'})

    def clean_exclusions(x):
        if isinstance(x, float):
            return ""
        x = x.replace('Carlos Ramon Ponce', 'Carlos Ponce')
        return ';'.join([x.strip().lower() for x in x.strip().replace('\n', ',').replace('.', ',').split(',')])

    #users
    users['conflicts'] = users.MindMatchExclude.map(clean_exclusions)
    users['fullname'] = ((users.NameFirst + ' ') + users.NameLast).str.lower()

    # Drop the one duplicated user
    users = users[~users.user_id.duplicated(keep='last')]
    users = users.reset_index(drop=True)

    abstracts = users.RepresentativeWork.values.tolist()

    # Go from concatenated abstracts to individual ones.
    abstract_index, all_abstracts = deconcatenate_abstracts(abstracts)

    # Encode each abstract separately.
    model = SentenceTransformer('all-mpnet-base-v2')
    individual_encodings = model.encode(all_abstracts, show_progress_bar=True)

    D = individual_encodings @ individual_encodings.T
    D = D.ravel()

    # We use a rule that a person's match to another is determined by their best matched abstracts,
    # rather than their average. Aggregate across all the abstracts shared between two people to 
    # find the max.
    first_index = np.array(abstract_index).reshape((-1, 1)) @ np.ones((1, len(abstract_index)))
    second_index = first_index.T

    first_index = first_index.ravel()
    second_index = second_index.ravel()

    first_abstract = np.arange(len(abstract_index)).reshape((-1, 1)) @ np.ones((1, len(abstract_index)))
    second_abstract = first_abstract.T

    first_abstract = first_abstract.ravel()
    second_abstract = second_abstract.ravel()

    df_individual_match = pd.DataFrame({'first_participant': pd.Series(first_index), 
                                        'second_participant': pd.Series(second_index), 
                                        'first_abstract': pd.Series(first_abstract), 
                                        'second_abstract': pd.Series(second_abstract), 
                                        'match': pd.Series(D.ravel())})

    df_idxmax = df_individual_match.groupby(['first_participant', 'second_participant']).idxmax()
    nums = df_idxmax.match

    # translate nums into left abstract, top abstract
    first_abstract, second_abstract = nums % len(all_abstracts), nums // len(all_abstracts)

    df_idxmax['first_abstract'] = first_abstract
    df_idxmax['second_abstract'] = second_abstract

    df_idxmax.to_pickle('data/transformed/abstract_indices.pkl')

    with open('data/transformed/all_abstracts.pkl', 'wb') as f:
        pickle.dump(all_abstracts, f)

    the_best = df_individual_match.groupby(['first_participant', 'second_participant']).max()

    # The result: a match matrix.
    M = the_best.match.values.reshape((len(abstracts), len(abstracts)))
    M = (1 - M) / 2.0

    np.save('data/transformed/match_matrix.npy', M)
    users.to_pickle('data/transformed/match_users_filled.pkl')



if __name__ == '__main__':
    main()