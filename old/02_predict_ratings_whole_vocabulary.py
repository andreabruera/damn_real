import argparse
import fasttext
import gensim
import logging
import numpy
import os
import pickle
import sklearn

from gensim.models import Word2Vec
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

from utils import read_args

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

args = read_args(mode='prediction')
args.model = 'fasttext'

file_path = os.path.join(
                         'data',
                         'Lancaster_sensorimotor_norms_for_39707_words.tsv',
                         )
assert os.path.exists(file_path)
relevant_keys = [
                 'Auditory.mean',
                 'Gustatory.mean',
                 'Haptic.mean',
                 'Olfactory.mean',
                 'Visual.mean',
                 'Foot_leg.mean',
                 'Hand_arm.mean', 
                 'Head.mean', 
                 'Mouth.mean', 
                 'Torso.mean'
                 ]

norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.strip().split('\t')
        if counter == 0:
            header = line.copy()
            counter += 1
            continue
        assert len(line) == len(header)
        marker = False
        for k in relevant_keys:
            try:
                assert float(line[header.index(k)]) < 10
            except AssertionError:
                #logging.info(line[0])
                marker = True
        if marker:
            continue
        if len(line[0].split()) == 1:
            norms[line[0].lower()] = list()
            for k in relevant_keys:
                val = float(line[header.index(k)])
                norms[line[0].lower()].append(val)

logging.info('number of annotated words retained for training: {}'.format(len(norms.keys())))

### reading frequencies
ratings = dict()
for corpus in ['opensubs', 'wac']:
    freqs = pickle.load(open(os.path.join('..', 'psychorpus', 'pickles', args.language, '{}_{}_word_freqs.pkl'.format(args.language, corpus)), 'rb'))
    for original_k, v in freqs.items():
        k = original_k.lower()
        ### frequencies are not lowercased
        if k not in ratings.keys():
            ratings[k] = v
        else:
            ratings[k] += v

logging.info('now reducing the vocabulary')
reduced_vocab = [k for k, v in ratings.items() if v > 50]
logging.info(len(reduced_vocab))

### loading fasttext model

if args.model == 'fasttext':
    logging.info('now loading fasttext')
    model = fasttext.load_model(os.path.join(
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
    model_vocabulary = model.words
if args.model == 'w2v':
    logging.info('now loading w2v')
    model = Word2Vec.load(os.path.join(
                                    '/',
                                    'import',
                                    'cogsci',
                                    'andrea',
                                    'dataset',
                                    'word_vectors',
                                    'en',
                                    'word2vec_en_opensubs+wac_param-mandera2017_min-count-50',
                                    'word2vec_en_opensubs+wac_param-mandera2017_min-count-50.model',
                                    )
                                    ).wv
    model_vocabulary = [w for w in model.vocab]

logging.info('now preparing the training data')
training_words = [w for w in norms.keys() if w in model_vocabulary]
training_input = [model[w] for w in training_words]
for vec in training_input:
    assert vec.shape == (300, )
training_target = [numpy.array(norms[w], dtype=numpy.float32) for w in training_words]
for vec in training_target:
    assert vec.shape == (len(relevant_keys), )

logging.info('now preparing the test data')
to_be_predicted_words = [w for w in reduced_vocab if w not in training_words and w in model_vocabulary]
to_be_predicted_input = [model[w] for w in to_be_predicted_words]
for vec in to_be_predicted_input:
    assert vec.shape == (300, )

logging.info('now training...')
ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
ridge.fit(training_input, training_target)
logging.info('...finally predicting and writing to file!')
predictions = ridge.predict(to_be_predicted_input)

out_folder = os.path.join('predictions')
os.makedirs(out_folder, exist_ok=True)

with open(os.path.join(out_folder, 'sensory_training_{}.tsv'.format(args.model)), 'w') as o:
    o.write('word\t')
    for k in relevant_keys:
        o.write('{}\t'.format(k))
    o.write('\n')
    for k, v in zip(training_words, training_target):
        o.write('{}\t'.format(k))
        for pred in v:
            o.write('{}\t'.format(pred))
        o.write('\n')

with open(os.path.join(out_folder, 'sensory_predicted_{}.tsv'.format(args.model)), 'w') as o:
    o.write('word\t')
    for k in relevant_keys:
        o.write('{}\t'.format(k))
    o.write('\n')
    for k, v in zip(to_be_predicted_words, predictions):
        o.write('{}\t'.format(k))
        for pred in v:
            o.write('{}\t'.format(pred))
        o.write('\n')

import pdb; pdb.set_trace()

### loading wac
if args.language == 'en':
    identifier = 'PukWaC'
elif args.language == 'it':
    identifier = 'itwac'
elif args.language == 'de':
    identifier = 'sdewac-v3-tagged'
wac_folder = os.path.join(args.corpora_path, args.language, '{}_smaller_files'.format(identifier))
assert os.path.exists(wac_folder)
### loading opensubs
opensubs_folder = os.path.join(args.corpora_path, args.language, 'opensubs_ready')
assert os.path.exists(opensubs_folder)
