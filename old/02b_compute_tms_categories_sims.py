import fasttext
import logging
import numpy
import os
import pickle
import re
import scipy

from tqdm import tqdm
from scipy import spatial

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

base = os.path.join('..', 'sfursat')
assert os.path.exists(base)

logging.info('now loading fasttext')
model = fasttext.load_model(os.path.join(
                                '/',
                                'import',
                                'cogsci',
                                'andrea',
                                'dataset',
                                'word_vectors',
                                'de',
                                'cc.de.300.bin',
                                )
                                )
model_vocabulary = model.words

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
for orig, cat in to_damage.items():
    print(orig)
    sims = dict()
    for k, v in tqdm(cn.items()):
        if k == orig:
            continue
        sim = 1 - scipy.spatial.distance.cosine(cn[cat], v)
        sims[k] = sim
    all_sims[orig] = sims

import pdb; pdb.set_trace()
