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

def bins_rsa_test(dataset, all_data, out_file, model):

    n_items = 32
    n_items = 8
    dataset_results = dict()

    all_words = relevant_dataset_words[dataset]
    #for w in words:
    #    assert w in model.vocab
    missing_words = [w for w in all_words if w not in model.vocab]
    print('missing {} words'.format(len(missing_words)))
    words = [w for w in all_words if w in model.vocab]
    bins = {'{}_{}'.format(case.split('.')[0], i) : list() for i in range(5) for case in relevant_keys}

    for case_i, case in enumerate(relevant_keys):
        print(case)
        case = case.split('.')[0]
        counter = 0
        for beg, end in tqdm([(0., 1.), (1, 2), (2, 3), (3, 4), (4, 5)]):
            bin_words = [w for w in words if norms[w][case_i]>=beg and norms[w][case_i]<=end]
            if len(bin_words) < n_items:
                #print('nan')
                bins['{}_{}'.format(case, counter)] = [numpy.nan for s in range(all_data[words[0]].shape[0])]
                counter += 1
                continue
            bin_results = list()
            for _ in range(100):
                iter_results = list()
                current_bin_words = random.sample(bin_words, k=n_items)
                current_bin_idxs = [all_words.index(w) for w in current_bin_words]

                sim_model = numpy.array([[1 - scipy.spatial.distance.cosine(model[k], model[k_two]) for k_two in current_bin_words if k!=k_two] for k in current_bin_words]).flatten()
                data = {k : numpy.array([[vec[i] for i in current_bin_idxs if all_words[i]!=k] for vec in v]) for k, v in all_data.items() if k in current_bin_words}
                for s in range(data[current_bin_words[0]].shape[0]):
                    brain = numpy.array([data[w][s, :] for w in current_bin_words]).flatten()
                    corr = scipy.stats.pearsonr(sim_model, brain)[0]
                    iter_results.append(corr)
                bin_results.append(iter_results)
            bin_results = numpy.average(bin_results, axis=0)
            #print(bin_results)
            bins['{}_{}'.format(case, counter)] = bin_results
            counter += 1

    with open(out_file, 'w') as o:
        o.write('bin\tresults\n')
        for k, v in bins.items():
            assert len(v) == data[current_bin_words[0]].shape[0]
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
                            'bins_rsa_{}_undamaged_{}_{}.results'.format(
                                   dataset, 
                                   args.model, 
                                   args.language,
                                   )
                            )
    bins_rsa_test(dataset, data, undamaged_file, model)

for dataset, data in datasets.items():

    ### loading models

    relu_bases=[
             #'50', 
             #'75', 
             #'90',
             '95',
             ] 
    sampling=[
             'random', 
             #'inverse', 
             #'pos',
             ]
    functions=[
             #'sigmoid', 
             #'raw', 
             #'exponential', 
             #'relu-raw-thresholded99', 
             #'relu-raw-thresholded90', 
             #'relu-raw-thresholded85', 
             #'relu-raw-thresholded95', 
             'relu-raw', 
             #'relu-exponential', 
             #'logarithmic', 
             #'relu-logarithmic', 
             #'relu-sigmoid', 
             #'relu-step',
             ]
    semantic_modalities = [
                        'auditory',
                        #'action',
                        ]
    for sem_mod in semantic_modalities:
        for func in functions:
            for relu_base in relu_bases:
                #if 'relu' not in func and relu_base != '50':
                #    continue
                for s in sampling:

                    args.function = func
                    args.relu_base = relu_base
                    args.sampling = s
                    args.semantic_modality = sem_mod

                    _, setup_info = prepare_input_output_folders(args, mode='plotting')
                    print(setup_info)

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
                        model_file = os.path.join(
                                            'models', 
                                            "word2vec_{}_damaged_{}_param-mandera2017_min-count-50.model".format(
                                                       args.language, 
                                                       setup_info)
                                            )
                        assert os.path.exists(model_file)
                        model = Word2Vec.load(os.path.join(
                                                        #'/',
                                                        #'import',
                                                        #'cogsci',
                                                        #'andrea',
                                                        #'dataset',
                                                        #'word_vectors',
                                                        #'en',
                                                        #'models',
                                                        model_file
                                                        )
                                                        ).wv
                    out_file = os.path.join(
                                     'brain_results', 
                                     'bins_rsa_{}_{}_{}_{}.results'.format(
                                     dataset,
                                     args.model, 
                                     args.language, 
                                     setup_info,
                                     )
                                     )
                    bins_rsa_test(dataset, data, out_file, model)
