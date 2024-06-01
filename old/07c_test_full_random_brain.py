import argparse
import fasttext
import gensim
import logging
import matplotlib
import numpy
import os
import pickle
import random
import scipy
import sklearn
import spacy

from gensim.models import Word2Vec
from matplotlib import font_manager, pyplot
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from tqdm import tqdm

from utils import prepare_input_output_folders, read_args

def rsa_test(dataset, data, out_file, model):
    dataset_results = dict()

    words = relevant_dataset_words[dataset]
    for w in words:
        assert w in model.vocab
    relevant_vecs = {w : model[w] for w in words}
    randomized_vecs = random.sample([v for v in relevant_vecs.values()], k=len(relevant_vecs.keys()))
    noisy_model = {k : v for k, v in zip(relevant_vecs.keys(), randomized_vecs)}

    sim_model = {k : [1 - scipy.spatial.distance.cosine(noisy_model[k], noisy_model[k_two]) for k_two in words] for k in words}

    ### leave-one-out
    for test_i, test_item in tqdm(enumerate(words)):
        item_results = list()
        training_input = [model[w] for w in words if w!=test_item]
        test_input = model[test_item]
        training_input = [numpy.delete(sim_model[w], test_i, axis=0) for w in words if w!=test_item]
        test_input = numpy.delete(sim_model[w], test_i, axis=0)
        all_training_target = [numpy.delete(data[w], test_i, axis=1) for w in words if w!=test_item]
        all_test_target = numpy.delete(data[w], test_i, axis=1)
        all_ranking_targets = [numpy.delete(data[w], test_i, axis=1) for w in words]
        for s in range(all_test_target.shape[0]):
            training_target = [brain[s, :] for brain in all_training_target]
            test_target = all_test_target[s, :]
            corr = scipy.stats.pearsonr(test_input, test_target)[0]
            item_results.append(corr)
        dataset_results[test_item] = item_results

    with open(out_file, 'w') as o:
        o.write('word\tresults\n')
        for k, v in dataset_results.items():
            o.write('{}\t'.format(k))
            for val in v:
                o.write('{}\t'.format(val))
            o.write('\n')

def ridge_test(dataset, data, out_file, model):
    dataset_results = dict()

    words = relevant_dataset_words[dataset]

    for w in words:
        assert w in model.vocab

    all_vectors = {w : model[w] for w in words}
    for vec in all_vectors.values():
        assert vec.shape == (300, )

    ### leave-one-out
    for test_i, test_item in tqdm(enumerate(words)):
        item_results = list()
        training_input = [all_vectors[w] for w in words if w!=test_item]
        test_input = all_vectors[test_item]
        all_training_target = [numpy.delete(data[w], test_i, axis=1) for w in words if w!=test_item]
        all_test_target = numpy.delete(data[w], test_i, axis=1)
        all_ranking_targets = [numpy.delete(data[w], test_i, axis=1) for w in words]
        for s in range(all_test_target.shape[0]):
            training_target = [brain[s, :] for brain in all_training_target]
            test_target = all_test_target[s, :]
            ranking_targets = [brain[s, :] for brain in all_ranking_targets]
            ridge = sklearn.linear_model.RidgeCV(alphas=(0.01, 0.1, 1, 10, 100, 1000, 10000))
            ridge.fit(training_input, training_target)
            prediction = ridge.predict([test_input])
            if prediction.shape[0] == 1:
                prediction = prediction[0]
            ### ranking
            corrs = [(real_w, scipy.stats.pearsonr(prediction, real)[0]) for real_w, real in zip(words, ranking_targets)]
            sorted_corrs = sorted(corrs, key=lambda item : item[1], reverse=True)
            sorted_words = [w[0] for w in sorted_corrs]
            rank = 1 - (sorted_words.index(test_item) / len(words))
            item_results.append(rank)
        dataset_results[test_item] = item_results

    with open(out_file, 'w') as o:
        o.write('word\tresults\n')
        for k, v in dataset_results.items():
            o.write('{}\t'.format(k))
            for val in v:
                o.write('{}\t'.format(val))
            o.write('\n')

# Using Helvetica as a font
font_folder = '/import/cogsci/andrea/dataset/fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

args = read_args(mode='results')

overall_keys = list()

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
overall_keys.extend(relevant_keys)

norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.strip().split('\t')
        if counter == 0:
            #logging.info(line)
            header = line.copy()
            counter += 1
            continue
        assert len(line) == len(header)
        word = line[0].lower()
        norms[word] = list()
        marker = False
        for k in relevant_keys:
            #try:
            #    assert float(line[header.index(k)]) <= 5
            #except AssertionError:
            #    marker = True
            if float(line[header.index(k)]) > 10:
                line[header.index(k)] = '.{}'.format(line[header.index(k)])
            assert float(line[header.index(k)]) < 10
        if marker:
            continue
        for k in relevant_keys:
            norms[word].append(float(line[header.index(k)]))

### reading datasets
dataset_words = {
            'fernandino1' : dict(),
            'fernandino2' : dict(),
            }
for d in dataset_words.keys():
    with open(os.path.join('data', '{}_words.txt'.format(d))) as i:
        delimiter = '\n' if d!='mitchell' else ', '
        words = [w.strip() for w in i.read().strip().split(delimiter)]
        dataset_words[d] = words

datasets = {
            'fernandino1' : dict(),
            'fernandino2' : dict(),
            }
for d in datasets.keys():
    folder = os.path.join('fernandino_brain_data', '{}_ALE_semantic_network'.format(d))
    assert os.path.exists(folder)
    for f in os.listdir(folder):
        with open(os.path.join(folder, f)) as i:
            for l_i, l in enumerate(i):
                current_word = dataset_words[d][l_i]
                if current_word in norms.keys():
                    line = l.strip().split('\t')
                    assert len(line) == len(dataset_words[d])
                    ### correcting vector
                    vector = [c for c, w in zip(line, dataset_words[d]) if w in norms.keys()]
                    if current_word not in datasets[d].keys():
                        datasets[d][current_word] = numpy.array(vector, dtype=numpy.float64)
                    else:
                        datasets[d][current_word] = numpy.row_stack([datasets[d][current_word], numpy.array(vector, dtype=numpy.float64)])

relevant_dataset_words = {k : [val for val in v if val in norms.keys()] for k, v in dataset_words.items()}

for k, v in datasets.items():
    for vec in v.values():
        assert vec.shape[-1] == len(relevant_dataset_words[k])

###

os.makedirs('brain_results', exist_ok=True)

### undamaged model
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

### checking words are all in the vocabulary
for dataset, data in datasets.items():
    undamaged_file = os.path.join(
                            'brain_results', 
                            'rsa_{}_undamaged_{}_{}.results'.format(
                                   dataset, 
                                   args.model, 
                                   args.language,
                                   )
                            )
    if not os.path.exists(undamaged_file):
        #ridge_test(dataset, data, undamaged_file, model)
        rsa_test(dataset, data, undamaged_file, model)

for dataset, data in datasets.items():

    ### adding noise into model

    ### loading models

    out_file = os.path.join(
                     'brain_results', 
                     'rsa_{}_{}_{}_full_random.results'.format(
                     dataset,
                     args.model, 
                     args.language, 
                     )
                     )
    rsa_test(dataset, data, out_file, model)
