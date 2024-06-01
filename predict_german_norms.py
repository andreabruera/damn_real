import multiprocessing
import numpy
import os
import pickle
import scipy
import sklearn

from scipy import spatial
from sklearn import linear_model
from tqdm import tqdm

from utils import read_ratings

print('now loading models...')

### loading aligned german fasttext
ft_de_file = os.path.join('pickles', 'ft_de_aligned.pkl')
if os.path.exists(ft_de_file):
    with open(ft_de_file, 'rb') as i:
        ft_de = pickle.load(i)
else:
    ft_de = dict()
    with open(os.path.join('..', '..', 'dataset', 'word_vectors', 'de', 'wiki.de.align.vec')) as i:
        for l_i, l in enumerate(i):
            line = l.split(' ')
            if l_i == 0:
                continue
            ft_de[line[0]] = numpy.array(line[1:], dtype=numpy.float64)
            assert ft_de[line[0]].shape == (300, )
    with open(ft_de_file, 'wb') as i:
        pickle.dump(ft_de, i)

### loading aligned english fasttext
ft_en_file = os.path.join('pickles', 'ft_en_aligned.pkl')
if os.path.exists(ft_en_file):
    with open(ft_en_file, 'rb') as i:
        ft_en = pickle.load(i)
else:
    ft_en = dict()
    with open(os.path.join('..', '..', 'dataset', 'word_vectors', 'en', 'wiki.en.align.vec')) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split(' ')
            if l_i == 0:
                continue
            ft_en[line[0]] = numpy.array(line[1:], dtype=numpy.float64)
    with open(ft_en_file, 'wb') as i:
        pickle.dump(ft_en, i)

print('loaded!')

lancaster_ratings = read_ratings(hand=True) 
counter = 0

### preparing german inputs
def sim_computer(ins):
    w_i = ins[0]
    w = ins[1]
    ft_en = ins[2]
    ft_de = ins[3]
    if w not in ft_en.keys():
        w = ''
        res = ''
    else:
        sims = list()
        for de_w, de_vec in tqdm(ft_de.items()):
            assert de_vec.shape == (300,)
            assert ft_en[w].shape == (300,)
            sim = 1 - scipy.spatial.distance.cosine(ft_en[w], de_vec)
            sims.append((de_w, sim))
        #res = numpy.average([ft_de[w] for w in [v[0] for v in sorted(sims, key=lambda item : item[1], reverse=True)][:10]], axis=0) 
        res = [v[0] for v in sorted(sims, key=lambda item : item[1], reverse=True)][:1000]
        print(res)
    with open(os.path.join('data', 'german_lancaster_predicted_missing_norms', '{}.tsv'.format(w)), 'w') as o:
        o.write('{}\t'.format(w))
        for p in res:
            o.write('{}\t'.format(p))
        o.write('\n')
    print(w_i)

out_f = os.path.join('data', 'german_lancaster_predicted_missing_norms')
os.makedirs(out_f, exist_ok=True)

with multiprocessing.Pool(processes=int(os.cpu_count())) as pool:
    sims = pool.map(sim_computer, [(w_i, w, ft_en, ft_de) for w_i, w in enumerate(lancaster_ratings.keys())])
    pool.terminate()
    pool.join()

'''

germ_sims = {k : v for k, v in sims.items() if k!=''}

model = sklearn.linear_model.RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
ws = list(lancaster_ratings.keys())
ks = list(lancaster_ratings.keys())
model.fit(
          [ft_en[w] for w in ws],
          [[lancaster_ratings[w][k] for k in ks] for w in ws]
          )
predictions = model.predict(
                            [germ_inputs[w] for w in ws]
                            )
'''
