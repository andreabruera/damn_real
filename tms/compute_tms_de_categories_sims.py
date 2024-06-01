import fasttext
import logging
import numpy
import os
import pickle
import re
import scipy

from scipy import spatial
from tqdm import tqdm

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

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

base = os.path.join('..', 'sfursat')
assert os.path.exists(base)
### now loading conceptnet
logging.info('now loading conceptnet')
with open(os.path.join(base, 'pickles', 'conceptnet_de.pkl'), 'rb') as o:
    cn = pickle.load(o)

### read dataset
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

### checking words are in conceptnet vocab
logging.info('now checking words are in conceptnet')
to_damage = dict()
for orig in set(full_dataset['category']):
    if orig == 'NA':
        continue
    if 'igkeiten' in orig:
        cat = 'süssigkeiten'
    elif orig == 'Autoteile':
        cat = 'autoteil'
    else:
        cat = orig.lower()
    if cat not in cn.keys():
        print(cat)
        continue
    to_damage[orig] = cat

all_sims = dict()
for orig, cat in tqdm(to_damage.items()):
    print(orig)
    sims = dict()
    for k, v in tqdm(cn.items()):
        if k == orig:
            continue
        sim = 1 - scipy.spatial.distance.cosine(cn[cat], v)
        sims[k] = sim
    all_sims[orig] = sims

with open(os.path.join('pickles', 'conceptnet_de_sims.pkl'), 'wb') as o:
    pickle.dump(all_sims, o)
