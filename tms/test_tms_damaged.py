import multiprocessing
import numpy
import os
import pickle
import random
import re
import scipy
import sklearn
import tqdm

from sklearn import linear_model
from tqdm import tqdm

from utils import bins_rsa_test, full_dataset_rsa_test, build_ppmi_vecs, read_fernandino_ratings, read_ratings, read_men, read_men_test, read_simlex, read_fernandino
from utf_utils import transform_german_word

def sims_damage(ctx_words, idxs):
    damaged_pmi_mtrx = numpy.copy(clean_mtrx)
    print(damage_type)
    for idx in idxs:
        ### rows
        damaged_pmi_mtrx[:, idx][idxs] = 0.
        ### columns
        damaged_pmi_mtrx[idx, :][idxs] = 0.
    vecs = {k : v for k, v in zip(ctx_words, damaged_pmi_mtrx)}
    print(len(idxs))
    for w_idx in idxs:
        w = ctx_words[w_idx]
        #print(w)
        for idx in idxs:
            assert vecs[w][idx] == 0.
    
    return vecs

### reading similarities
with open(os.path.join('pickles', 'conceptnet_de_sims.pkl'), 'rb') as i:
    cn_sims = pickle.load(i)

### read dataset
base = os.path.join('..', 'sfursat')
assert os.path.exists(base)
print('now loading TMS dataset')
lines = list()
with open(os.path.join(base, 'data', 'all_tasks.tsv')) as i:
    for l_i, l in enumerate(i):
        l = re.sub(r'\'|\"', r'', l)
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            full_dataset = {h : list() for h in header}
            continue
        for val, h in zip(line, header):
            full_dataset[h].append(val)
total_rows = l_i

rts = {int(sub) : {
                     task : {
                         cond : {
                                #cat : list() for cat in set(full_dataset['item'])
                                } 
                         for cond in set(full_dataset['cond'])
                         } 
                     for task in set(full_dataset['task'])
                     } 
                     for sub in set(full_dataset['participant'])
                     }
fluencies = {int(sub) : {
                     task : {
                         cond : {
                                #cat : list() for cat in set(full_dataset['item'])
                                } 
                         for cond in set(full_dataset['cond'])
                         } 
                     for task in set(full_dataset['task'])
                     } 
                     for sub in set(full_dataset['participant'])
                     }

sim_results = {'dual' : dict(), 'sham': dict(), 'IFG': dict(), 'preSMA' : dict()}
ml_results = {'dual' : dict(), 'sham': dict(), 'IFG': dict(), 'preSMA' : dict()}
cats = set()

for row in tqdm(range(total_rows)):
    sub = int(full_dataset['participant'][row])
    task = full_dataset['task'][row]
    cond = full_dataset['cond'][row]
    cat = full_dataset['item'][row]
    if cat not in cn_sims.keys():
        print(cat)
        continue
    rt = float(full_dataset['rt'][row])
    word = full_dataset['response'][row].strip()
    if 'sem' in task:
        cats.add(cat)
    if cat not in rts[sub][task][cond].keys():
        rts[sub][task][cond][cat] = list()
        fluencies[sub][task][cond][cat] = list()
    rts[sub][task][cond][cat].append(rt)
    fluencies[sub][task][cond][cat].append(word)

### metrics vs difficulty
difficulties = dict()
with open(os.path.join(base, 'data', 'category_ranking.tsv')) as i:
    for l_i, l in enumerate(i):
        if l_i==0:
            continue
        line = l.strip().split('\t')
        difficulties[line[0]] = float(line[2])
import pdb; pdb.set_trace()

### reading files
### loading undamage model
corpus = 'joint'
min_count = 10
win_size = 4
marker = 'pmi_smooth75'
full_corpus = '{}_coocs_uncased_min_{}_win_{}'.format(corpus, min_count, win_size)
marker = 'pmi_smooth75'
with open(os.path.join(
                       'pickles', 'de', corpus, 
                       'de_{}_uncased_word_pos.pkl'.format(corpus),
                       ), 'rb') as i:
    pos = pickle.load(i)
with open(os.path.join(
                       'pickles', 'de', corpus, 
                       'de_{}_uncased_vocab_min_{}.pkl'.format(corpus, min_count),
                       ), 'rb') as i:
    vocab = pickle.load(i)
out_f = os.path.join(
                     'damaged_pickles', 'de', full_corpus, marker, 
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
                  ]

with tqdm() as counter:
    for dam in damage_amounts:
        for cat in cats:
            print(cat)
            corr_cat = transform_german_word(cat)
            corr_cat = set([w for w in corr_cat])
            #corr_cat = [cat]
            cat_vec = dict()
            cat_vec['damaged'] = numpy.average([damaged_ft.get_word_vector(w) for w in corr_cat], axis=0)
            cat_vec['undamaged'] = numpy.average([undamaged_ft.get_word_vector(w) for w in corr_cat], axis=0)
            ### similarities first
            for sub, sub_data in fluencies.items():
                for task, task_data in sub_data.items():
                    if 'sem' not in task:
                        continue
                    #print(task)
                    for cond, cond_data in task_data.items():
                        if cat not in cond_data.keys():
                            continue
                        data = cond_data[cat]
                        cat_results = {'damaged' : list(), 'undamaged' : list()}

                        for word in data:
                            corr_word = transform_german_word(word)
                            corr_toks = set([w for c_w in corr_word for w in c_w.split()])
                            #corr_toks = [word]
                            for ft_type, ft in [('damaged', damaged_ft), ('undamaged', undamaged_ft)]:
                                corr_vec = numpy.average([ft.get_word_vector(w) for w in corr_toks], axis=0)
                                sim = 1 - scipy.spatial.distance.cosine(corr_vec, cat_vec[ft_type])
                                cat_results[ft_type].append(sim)
                        if cat not in sim_results[cond].keys():
                            sim_results[cond][cat] = cat_results
                        else:
                            for k, v in cat_results.items():
                                sim_results[cond][cat][k].extend(v)
                                counter.update(1)

with open('TMS_sims_results.txt', 'w') as o:
    for typ in ['damaged', 'undamaged']:
        o.write('average similarities between {} model and words produced\n'.format(typ))
        o.write('\n')
        for_diff = dict()
        for cond, cond_data in sim_results.items():
            sims = [val for v in cond_data.values() for val in v[typ]]
            for_diff[cond] = sims
            avg = numpy.average(sims)
            #print([typ, cond, round(avg, 4)])
            o.write(str([typ, cond, round(avg, 4)]))
            o.write('\n')
        o.write('\n')
        #print('\n')
        o.write('p-values of difference between conditions for {} model and words produced\n'.format(typ))
        o.write('\n')
        mrkr = list()
        for cond_one, sims_one in for_diff.items():
            for cond_two, sims_two in for_diff.items():
                if cond_one == cond_two:
                    continue
                key = sorted([cond_one, cond_two])
                if key in mrkr:
                    continue
                mrkr.append(key)
                diff = scipy.stats.ttest_ind(sims_one, sims_two).pvalue
                o.write(str([cond_one, cond_two, round(diff, 5)]))
                o.write('\n')
        #print('\n\n')
        o.write('\n\n')


rankings = rank_words(lancaster_ratings, ctx_words, ranking_modes, to_be_damaged, damage_amounts)

for rat, rat_dict in r_dict.items():
    for damage_amount, idxs in rat_dict.items():
        model_results = list()
        out = os.path.join(
                           'results', 
                           'tms', 
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
            vecs = sims_damage(ctx_words, idxs)

        model_results.append((area, full_dataset_rsa_test(ins)))
        with open(out_file, 'w') as o:
            for area, l in model_results:
                o.write('{}\t'.format(area))
                for val in l:
                    o.write('{}\t'.format(val))
                o.write('\n')
