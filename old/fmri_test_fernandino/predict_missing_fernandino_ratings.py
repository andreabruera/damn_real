import fasttext
import os
import pickle
import sklearn

from sklearn import linear_model

from utils import read_ratings

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
lancaster_ratings = read_ratings(hand=True) 
missing_norms = [w for w in ctx_words if w not in lancaster_ratings.keys()]
assert len(missing_norms) != 0

ft = fasttext.load_model(os.path.join(
                                '/',
                                'import',
                                'cogsci',
                                'andrea',
                                'dataset',
                                'word_vectors',
                                'en',
                                'cc.en.300.bin',
                                )
                                )

keys = sorted(lancaster_ratings['dog'].keys())
words_targets = [(w, [v[k] for k in keys]) for w, v in lancaster_ratings.items()]
model = sklearn.linear_model.RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
model.fit(
          [ft[w] for w, v in words_targets],
          [v for w, v in words_targets]
          )
predictions = model.predict(
                            [ft[w] for w in missing_norms]
                            )
with open(os.path.join('data', 'fernandino_predicted_missing_norms.tsv'), 'w') as o:
    o.write('word\t')
    for k in keys:
        o.write('{}\t'.format(k))
    o.write('\n')
    for w, preds in zip(missing_norms, predictions):
        o.write('{}\t'.format(w))
        for p in preds:
            o.write('{}\t'.format(p))
        o.write('\n')
