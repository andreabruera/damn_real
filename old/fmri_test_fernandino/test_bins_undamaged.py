import multiprocessing
import numpy
import os
import pickle
import random
import scipy
import tqdm

from tqdm import tqdm

from utils import bins_rsa_test, build_ppmi_vecs, read_fernandino_ratings, read_ratings, read_men, read_men_test, read_simlex, read_fernandino

selected_areas = [
                  'semantic_network',
                  ###
                  'L_IFG',
                  'L_inferiorparietal',
                  'L_supramarginal',
                  'L_middletemporal',
                  'AMTG',
                  'PMTG',
                  'L_precuneus',
                  ###
                  #'AMTG',
                  #'L_caudalmiddlefrontal',
                  #'L_lateralorbitofrontal',
                  #'L_superiorfrontal',
                  #'ATL',
                  #'L_middletemporal',
                  #'R_IFG',
                  #'R_lateralorbitofrontal',
                  #'R_precuneus',
                  #'R_ATL',
                  ]
### loading undamage model
corpus = 'opensubs'
min_count = 10
win_size = 4
marker = 'pmi_smooth75'
full_corpus = '{}_coocs_uncased_min_{}_win_{}'.format(corpus, min_count, win_size)
marker = 'pmi_smooth75'
out_f = os.path.join(
                     'damaged_pickles', 'en', full_corpus, marker, 
                       )
assert os.path.exists(out_f)
with open(os.path.join(
                        out_f,
                       'words_rows_cols.pkl',
                       ), 'rb') as i:
    ctx_words = pickle.load(i)
with open(os.path.join(
                       out_f,
                       'undamaged_coocs.pkl',
                       ), 'rb') as i:
    mtrx = pickle.load(i)
assert mtrx.shape == (len(ctx_words), len(ctx_words))

global ratings
ratings = {k : v for k, v in read_fernandino_ratings().items() if k in ctx_words}
unit = 25
assert 100 % unit == 0
global splits
splits = [(i*0.01, (i+unit)*0.01) for i in range(0, 100, unit)]

seed = 12
global relevant_keys
relevant_keys = [
            'Audition',
            'Practice',
            'UpperLimb',
            'Touch',
            'Vision',
            'Taste',
            'Smell',
            'Sound',
            ]
global models
#models = {k : {w : vals[k] for w, vals in ratings.items()} for k in relevant_keys}
models = dict()
models[(marker, corpus)] = {w : mtrx[w_i] for w_i, w in enumerate(ctx_words)}
### dummy variables
global vocab
vocab = {k : 1 for k in ratings.keys()}
global pos
pos = {k : 1 for k in ratings.keys()}

print('now loading the models...')
all_words = dict()
all_data = dict()
for mode in [
            'averaged_sub_mtrxs', 
            'individual_sub_mtrxs',
            ]:
    if mode == 'averaged_sub_mtrxs':
        words, _, data = read_fernandino(vocab, pos, return_dict=True, avg_subjects=True)
    elif mode == 'individual_sub_mtrxs':
        words, _,data = read_fernandino(vocab, pos, return_dict=True, avg_subjects=False)
    all_words[mode] = words.copy()
    all_data[mode] = data.copy()
    del words
    del data
print('loaded!')
inputs = list()
print('now preparing inputs...')
for mode, data in all_data.items():
    for dataset, brain_data in data.items():
        for area, area_data in brain_data.items():
            if area not in selected_areas:
                continue
            for variable, vecs in models.items():
                if type(variable) == tuple:
                    marker = variable[0]
                    variable = '{}_{}_{}'.format(corpus, min_count, win_size)
                else:
                    marker = 'raw_score'
                ### undamaged
                out = os.path.join(
                                   'results', 
                                   'fernandino{}'.format(dataset), 
                                   'bins',
                                   mode,
                                   'undamaged',
                                   area,
                                   variable,
                                   marker,
                                   )
                os.makedirs(out, exist_ok=True)
                #print(out)
                out_file = os.path.join(
                                out, 
                                'undamaged.results'
                                )
                #inputs.append([mode, dataset, area, variable, out_file])
                inp = [
                        all_words[mode][dataset],
                        all_data[mode][dataset][area],
                        vecs,
                        out_file,
                        splits,
                        relevant_keys, 
                        ratings,
                        ]
                bins_rsa_test(inp)
'''
print('ready!')
if debugging:
    #bins_rsa_test(words[dataset], fern_ratings, vecs, area_data, splits, out_file)
    for inp in inputs:
else:
    with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
        pool.map(bins_rsa_test, inputs)
        pool.terminate()
        pool.join()
'''
