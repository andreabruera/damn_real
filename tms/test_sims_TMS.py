import fasttext
import logging
import numpy
import os
import re
import scipy

from scipy import spatial, stats
from tqdm import tqdm

from utf_utils import transform_german_word

### read dataset
base = os.path.join('..', 'sfursat')
assert os.path.exists(base)
logging.info('now loading TMS dataset')
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

with tqdm() as counter:
    for cat in cats:
        print(cat)
        damaged_ft_file = os.path.join(ft_folder, '{}_wac_subs_for_fasttext.bin'.format(cat))
        if not os.path.exists(damaged_ft_file):
            continue
        damaged_ft = fasttext.load_model(damaged_ft_file)
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
