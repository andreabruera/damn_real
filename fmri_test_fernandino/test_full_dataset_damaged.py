import multiprocessing
import numpy
import os
import pickle
import random
import scipy
import sklearn
import tqdm

from sklearn import linear_model
from tqdm import tqdm

from utils import bins_rsa_test, full_dataset_rsa_test, build_ppmi_vecs, read_fernandino_ratings, read_ratings, read_men, read_men_test, read_simlex, read_fernandino

def rank_words(ratings, ctx_words, ranking_modes, to_be_damaged, damage_amounts):
    print('missing words: {}'.format([w for w in ctx_words if w not in ratings.keys()]))
    rankings = dict()
    for ranking_mode in ranking_modes:
        rankings[ranking_mode] = dict()
        for rat in to_be_damaged:
            rankings[ranking_mode][rat] = dict()
            ### here we use sensorimotor ratings
            if ranking_mode == 'absolute_val':
                sorted_ws = sorted([(w, v[rat]) for w, v in ratings.items() if w in ctx_words], key=lambda item: item[1])
            elif ranking_mode == 'dominance':
                curr_doms = [(w, numpy.average([v[rat]-v[rat_two] for rat_two in v.keys() if rat!=rat_two])) for w, v in ratings.items() if w in ctx_words]
                sorted_ws = sorted(curr_doms, key=lambda item: item[1])
                #print(sorted_ws)
            elif ranking_mode == 'dominance_and_val':
                curr_complexes = [(w, v[rat]+numpy.average([v[rat]-v[rat_two] for rat_two in v.keys() if rat!=rat_two])) for w, v in ratings.items() if w in ctx_words]
                sorted_ws = sorted([(w, curr_complexes) for w, v in ratings.items() if w in ctx_words], key=lambda item: item[1])
            for damage_amount in damage_amounts: 
                percent = int(len(sorted_ws)*damage_amount)
                lim_ctx_words = [w for w, val in sorted_ws[-percent:]]
                idxs = [ctx_words.index(w) for w in lim_ctx_words]
                assert len(idxs) > 0
                rankings[ranking_mode][rat][damage_amount] = idxs
    return rankings

def residualize(ctx_words, idxs, variable):

    vecs = {w : residualized_mtrx[variable][w_i] if w_i in idxs else clean_mtrx[w_i] for w_i, w in enumerate(ctx_words)}
    
    return vecs

def pow_rand_damage(damage_type, ctx_words, damage_mtrx, idxs):
    if 'anywhere' in damage_type:
        random.seed(12)
        idxs = random.sample(range(len(ctx_words)), k=len(idxs))
    damaged_pmi_mtrx = numpy.copy(clean_mtrx)
    print(damage_type)
    if 'pow' in damage_type:
        if damage_type == 'pow_0.':
            for idx in idxs:
                ### rows
                damaged_pmi_mtrx[:, idx][idxs] = 0.
                ### columns
                damaged_pmi_mtrx[idx, :][idxs] = 0.
        else:
            for idx in idxs:
                vals = damage_mtrx[:, idx][idxs]
                ### rows
                damaged_pmi_mtrx[:, idx][idxs] = vals 
                ### columns
                damaged_pmi_mtrx[idx, :][idxs] = vals 
    elif 'rand' in damage_type:
        random.seed(seed)
        for idx in idxs:
            vals = numpy.array(random.sample(damage_mtrx[:, idx].tolist(), k=len(idxs)))
            ### rows
            damaged_pmi_mtrx[:, idx][idxs] = vals
            ### columns
            damaged_pmi_mtrx[idx, :][idxs] = vals
    else:
        raise RuntimeError('not yet implemented')
    vecs = {k : v for k, v in zip(ctx_words, damaged_pmi_mtrx)}
    print(len(idxs))
    for w_idx in idxs:
        w = ctx_words[w_idx]
        #print(w)
        for idx in idxs:
            assert vecs[w][idx] == 0.
    
    return vecs


### reading files
### loading undamage model
corpus = 'opensubs'
min_count = 10
win_size = 4
marker = 'pmi_smooth75'
full_corpus = '{}_coocs_uncased_min_{}_win_{}'.format(corpus, min_count, win_size)
marker = 'pmi_smooth75'
with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_uncased_word_pos.pkl'.format(corpus),
                       ), 'rb') as i:
    pos = pickle.load(i)
with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_uncased_vocab_min_{}.pkl'.format(corpus, min_count),
                       ), 'rb') as i:
    vocab = pickle.load(i)
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
    clean_mtrx = pickle.load(i)
assert clean_mtrx.shape == (len(ctx_words), len(ctx_words))
powers = [
          #0.05, 
          #0.1,
          #0.25, 
          #0.5, 
          #0.75,
          ]



mtrxs = dict()
'''
for r_c in [
            'rand',
            ]:
    with open(os.path.join(
                               out_f,
                               '{}_rows.pkl'.format(r_c),
                           ), 'rb') as i:
        mtrxs[r_c] = pickle.load(i)
### power only
for power in powers:
    with open(os.path.join(
                           out_f,
                           'pow_{}.pkl'.format(power),
                           ), 'rb') as i:
        mtrxs['pow_{}'.format(power)] = pickle.load(i)
mtrxs['residualization'] = dict()
'''
mtrxs['pow_0.'] = dict()

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

debugging = False
ratings = read_fernandino_ratings()
lancaster_ratings = read_ratings(hand=True) 
missing_norms = [w for w in ctx_words if w not in lancaster_ratings.keys()]
assert len(missing_norms) == 0

to_be_damaged = [
                 'auditory',
                 'hand_arm',
                  ]
damage_amounts = [
                  0.05, 
                  0.1, 
                  0.2,
                  0.3,
                  0.4,
                  0.5, 
                  0.6,
                  0.7,
                  0.8, 
                  0.9,
                  0.95,
                  0.99
                  ]

### doing the word rankings first
### damaged
ranking_modes = [
                 'dominance', 
                 'absolute_val', 
                 #'dominance_and_val',
                 ]
rankings = rank_words(lancaster_ratings, ctx_words, ranking_modes, to_be_damaged, damage_amounts)

'''
### residualizing the matrix
print('now residualizing...')
residualized_mtrx = {k : clean_mtrx.copy() for k in to_be_damaged}
step = int(len(ctx_words)/10)
all_idxs = [w_i for w_i, w in enumerate(ctx_words) if w in lancaster_ratings.keys()]
print('number of words missing from the ratings: {}'.format(len(ctx_words)-len(all_idxs)))
for k in to_be_damaged:
    print(k)
    f = os.path.join('damaged_pickles', 'residualized_{}.pkl'.format(k))
    if os.path.exists(f):
        with open(f, 'rb') as i:
            residualized_mtrx[k] = pickle.load(i)
    else:

        split_points = list(range(0, len(ctx_words), step))[:-1]
        for start in split_points:
            train_idxs = all_idxs[0:start] + all_idxs[start+step:]
            if start == split_points[-1]:
                test_idxs = all_idxs[start:]
            else:
                test_idxs = all_idxs[start:start+step]
            model = sklearn.linear_model.LinearRegression()
            model.fit(
                      numpy.array([[lancaster_ratings[ctx_words[idx]][k]] for idx in train_idxs]), 
                      clean_mtrx[train_idxs],
                      )
            preds = model.predict(
                                  numpy.array([[lancaster_ratings[ctx_words[idx]][k]] for idx in test_idxs])
                                  )
            assert preds.shape[0] == len(test_idxs)
            for idx_i, idx in enumerate(test_idxs):
                residualized_mtrx[k][idx] = clean_mtrx[idx] - preds[idx_i]
        with open(f, 'wb') as o:
            pickle.dump(residualized_mtrx, o)
'''

modes = [
        'averaged_sub_mtrxs', 
        'individual_sub_mtrxs',
        ]

### loading data
global all_data
global all_words

all_data = dict()
all_words = dict()

for mode in modes:
    if mode == 'averaged_sub_mtrxs':
        words, _, data = read_fernandino(vocab, pos, return_dict=True, avg_subjects=True)
    elif mode == 'individual_sub_mtrxs':
        words, _,data = read_fernandino(vocab, pos, return_dict=True, avg_subjects=False)
    all_words[mode] = words.copy()
    all_data[mode] = data.copy()
    del words
    del data

print('preparing inputs...')
models = dict()
inputs = list()
for dataset in [1, 2]:
    for mode in modes:
        ### damages
        for damage_type, damage_mtrx in mtrxs.items():
            ###
            for ranking_mode, r_dict in rankings.items():
                for rat, rat_dict in r_dict.items():
                    for damage_amount, idxs in rat_dict.items():
                        model_results = list()
                        out = os.path.join(
                                           'results', 
                                           'fernandino{}'.format(dataset), 
                                           'full_dataset',
                                           mode,
                                           '{}_{}_{}_{}'.format(
                                                rat, 
                                                damage_type, 
                                                damage_amount, 
                                                ranking_mode,
                                                ),
                                           )
                        os.makedirs(out, exist_ok=True)
                        print(out)
                        out_file = os.path.join(
                                        out, 
                                        '{}.results'.format(
                                           '{}_{}_{}_{}'.format(marker, corpus, min_count, win_size),
                                           )
                                        )
                        if 'residualization' in damage_type:
                            vecs = residualize(ctx_words, idxs, rat)
                        else:
                            vecs = pow_rand_damage(damage_type, ctx_words, damage_mtrx, idxs)
                        for area in selected_areas:
                            ins = [
                                    all_words[mode][dataset],
                                    all_data[mode][dataset][area],
                                    vecs, 
                                    ]
                            model_results.append((area, full_dataset_rsa_test(ins)))
                        with open(out_file, 'w') as o:
                            for area, l in model_results:
                                o.write('{}\t'.format(area))
                                for val in l:
                                    o.write('{}\t'.format(val))
                                o.write('\n')
