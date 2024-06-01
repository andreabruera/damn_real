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

# Using Helvetica as a font
font_folder = '/import/cogsci/andrea/dataset/fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

args = read_args(mode='results')

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

overall_keys = list()

relevant_keys = ['Conc.M']
overall_keys.extend(relevant_keys)
file_path = os.path.join(
                         'data',
                         'brysbaert_conc.tsv',
                         )
assert os.path.exists(file_path)
norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.replace(',', '.').strip().split('\t')
        if counter == 0:
            header = line.copy()
            counter += 1
            continue
        assert len(line) == len(header)
        word = line[0].lower()
        #if word in ratings.keys():
        if len(word.split()) == 1:
            for k in relevant_keys:
            #if ratings[word] > 1000:
                norms[word] = [line[header.index(k)]]

logging.info(len(norms))

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

#norms = dict()
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
        #if word in ratings.keys():
        #    if len(word.split()) == 1:
        #        if ratings[word] > 1000:
        #            norms[word] = line[1:]
        ### checking because some lines contain errors
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

        if word in norms.keys():
            for k in relevant_keys:
            #if ratings[word] > 1000:
                norms[word].append(line[header.index(k)])
            #logging.info(overall_keys)
            #logging.info(norms[word])

'''
file_path = os.path.join(
                         'data',
                         #'lynott_perceptual.tsv',
                         'binder_ratings.tsv',
                         )
assert os.path.exists(file_path)
relevant_keys = [
        'Audition', 'Loud', 'Low', 'High','Sound', 'Music', 'Speech'
                 ]
overall_keys.extend(relevant_keys)

#norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.replace(',', '.').strip().split('\t')
        if counter == 0:
            #header = line.copy()[1:]
            header = line.copy()
            counter += 1
            continue
        #assert len(line[1:]) == len(header)
        word = line[1].lower()
        #if word in ratings.keys():
            #if len(word.split()) == 1:
                #if ratings[word] > 1000:
                #norms[word] = line[1:]
        if word in norms.keys():
            for k in relevant_keys:
            #if ratings[word] > 1000:
                norms[word].append(line[header.index(k)])

logging.info(len(norms))

### lynott
file_path = os.path.join(
                         'data',
                         'lynott_perceptual.tsv',
                         #'binder_ratings.tsv',
                         )
assert os.path.exists(file_path)

relevant_keys =  [
                 'Auditory_mean',
                 'Gustatory_mean',
                 'Haptic_mean',
                 'Olfactory_mean',
                 'Visual_mean',
                 ]
overall_keys.extend(relevant_keys)
#norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.replace(',', '.').strip().split('\t')
        if counter == 0:
            #header = line.copy()[1:]
            header = line.copy()
            counter += 1
            continue
        #assert len(line[1:]) == len(header)
        word = line[1].lower()
        #if word in ratings.keys():
            #if len(word.split()) == 1:
                #if ratings[word] > 1000:
                #norms[word] = line[1:]
        if word in norms.keys():
            for k in relevant_keys:
            #if ratings[word] > 1000:
                norms[word].append(line[header.index(k)])
#logging.info(len(norms))

'''

os.makedirs('results', exist_ok=True)

norms = {k : v for k,v in norms.items() if len(v)==len(overall_keys)}
logging.info(len(norms))

### undamaged model
undamaged_file = os.path.join('results', 'undamaged_{}_{}.results'.format(args.model, args.language,))
if not os.path.exists(undamaged_file):
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

        all_words = [w for w in norms.keys() if w in model_vocabulary]

        all_vectors = [model[w] for w in all_words]
        for vec in all_vectors:
            assert vec.shape == (300, )

        all_targets = [numpy.array(norms[w], dtype=numpy.float32) for w in all_words]
        for vec in all_targets:
            #assert vec.shape == (7, )
            assert vec.shape == (len(overall_keys), )
            #assert vec.shape == (5, )
            #assert vec.shape == (1, )

        ### 100 times random combinations
        twenty = int(len(all_words)*0.2)

        results = {k : list() for k in overall_keys}
        logging.info('now training/testing...')
        if args.debugging:
            iterations = 2
        else:
            iterations = 100
        for i in tqdm(range(iterations)):
            test_items = random.sample(all_words, k=twenty)
            training_input = [all_vectors[v_i] for v_i, v in enumerate(all_words) if v not in test_items]
            training_target = [all_targets[v_i] for v_i, v in enumerate(all_words) if v not in test_items]
            test_input = [all_vectors[all_words.index(v)] for v in test_items]
            test_target = [all_targets[all_words.index(v)] for v in test_items]
            ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
            #ridge = Ridge()
            ridge.fit(training_input, training_target)
            predictions = ridge.predict(test_input)
            for key_i, key in enumerate(overall_keys):
                ### computing correlations
                real = [target[key_i] for target in test_target]
                preds = [pred[key_i] for pred in predictions]
                corr = scipy.stats.pearsonr(real, preds)[0]
                results[key].append(corr)
        with open(undamaged_file, 'w') as o:
            o.write('Pearson correlation results for Brysbaert concreteness and perceptual strength norms (80-20 splits in 100 iterations of monte-carlo cross-validation) for the undamaged {} {}\n'.format(args.model, args.language))
            o.write('number of words retained: {} out of {}\n'.format(len(all_words), len(norms.keys())))
            for k, v in results.items():
                for val in v:
                    o.write('{}\t{}\n'.format(k, val))

### loading models

relu_bases=[
         #'50', 
         '75', 
         #'90',
         #'95',
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
         #'relu-raw-thresholded95', 
         #'relu-raw-thresholded85', 
         'relu-raw', 
         'relu-exponential', 
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
            if 'relu' not in func and relu_base != '50':
                continue
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
                    model_vocabulary = [w for w in model.vocab]

                    logging.info('now preparing the training data')

                    all_words = [w for w in norms.keys() if w in model_vocabulary]

                    all_vectors = [model[w] for w in all_words]
                    for vec in all_vectors:
                        assert vec.shape == (300, )

                    all_targets = [numpy.array(norms[w], dtype=numpy.float32) for w in all_words]
                    for vec in all_targets:
                        #assert vec.shape == (7, )
                        assert vec.shape == (len(overall_keys), )
                        #assert vec.shape == (5, )
                        #assert vec.shape == (1, )

                    ### 100 times random combinations
                    twenty = int(len(all_words)*0.2)

                    results = {k : list() for k in overall_keys}
                    logging.info('now training/testing...')
                    if args.debugging:
                        iterations = 2
                    else:
                        iterations = 100
                    for i in tqdm(range(iterations)):
                        test_items = random.sample(all_words, k=twenty)
                        training_input = [all_vectors[v_i] for v_i, v in enumerate(all_words) if v not in test_items]
                        training_target = [all_targets[v_i] for v_i, v in enumerate(all_words) if v not in test_items]
                        test_input = [all_vectors[all_words.index(v)] for v in test_items]
                        test_target = [all_targets[all_words.index(v)] for v in test_items]
                        ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
                        #ridge = Ridge()
                        ridge.fit(training_input, training_target)
                        predictions = ridge.predict(test_input)
                        for key_i, key in enumerate(overall_keys):
                            ### computing correlations
                            real = [target[key_i] for target in test_target]
                            preds = [pred[key_i] for pred in predictions]
                            corr = scipy.stats.pearsonr(real, preds)[0]
                            results[key].append(corr)
                    out_file = '{}_{}_{}'.format(
                                     args.model, 
                                     args.language, 
                                     setup_info,
                                     )
                    with open(os.path.join(
                                     'results', 
                                     '{}.results'.format(out_file)
                                     ), 'w') as o:
                        o.write('Pearson correlation results '\
                                'for Brysbaert concreteness and '\
                                'perceptual strength norms '\
                                '(80-20 splits in 100 iterations of '\
                                'monte-carlo cross-validation) '\
                                'for {}, {}, {}\n'.format(
                                    args.model, 
                                    args.language, 
                                    setup_info.replace('_', ' ')
                                    )
                                )
                        o.write('number of words retained: '\
                                '{} out of {}\n'.format(
                                                 len(all_words), 
                                                 len(norms.keys())
                                                 )
                                )
                        for k, v in results.items():
                            o.write('{}\t{}\n'.format(k, numpy.average(v)))

                    data = [
                            ('undamaged_w2v_en.results', 'undamaged'),
                            #('w2v_en_auditory_relu-raw-thresholded9095_random.results', 'damaged'),
                            ]
                    plot_results = dict()

                    for file_name, key in data:
                        ### reading unadamaged scores
                        plot_results[key] = dict()
                        with open(os.path.join('results', file_name)) as i:
                            counter = 0
                            for l in i:
                                if counter < 2:
                                    counter += 1
                                    continue
                                line = l.strip().split('\t')
                                try:
                                    plot_results[key][line[0]].append(float(line[1]))
                                except KeyError:
                                    plot_results[key][line[0]] = [float(line[1])]
                    plot_results['damaged'] = results.copy()

                    x_labels = list(plot_results[key].keys())
                    ### sorting x labels
                    to_be_sorted = [(abs(numpy.average(plot_results['undamaged'][k])-numpy.average(plot_results['damaged'][k])), k) for k in x_labels]
                    x_labels = [v[1] for v in sorted(to_be_sorted, key=lambda item : item[0], reverse=True)]
                    colors = {
                              'undamaged' : 'teal',
                              'damaged' : 'goldenrod',
                              }
                    scatters = {
                              'undamaged' : 'mediumturquoise',
                              'damaged' : 'darkgoldenrod',
                              }
                    corrections = {
                              'undamaged' : -.2,
                              'damaged' : .2,
                              }

                    fig, ax = matplotlib.pyplot.subplots(
                                                figsize=(20, 10), 
                                                constrained_layout=True,
                                                )
                    '''
                    for k, v in results.items():
                        bar_ys = [numpy.average(v[key]) for key in x_labels]
                        ax.bar(
                                [x+corrections[k] for x in range(len(bar_ys))], 
                                height=bar_ys, 
                                color=colors[k], 
                                label=k,
                                alpha=0.8,
                                edgecolor='lightgrey',
                                width=.3,
                                )
                        ### scatters
                        ax.scatter(
                                   [x_i+corrections[k]+(random.choice(range(-5,5))*0.025) for x_i, x in enumerate(x_labels) for val in v[x]],
                                   [val for x in x_labels for val in v[x]],
                                   color=scatters[k],
                                   edgecolors='white',
                                   )
                    '''
                    bar_ys = [numpy.average(plot_results['undamaged'][key])-numpy.average(plot_results['damaged'][key]) for key in x_labels]
                    ax.bar(
                            #[
                            #x+corrections[k] for x in range(len(bar_ys))], 
                            range(len(bar_ys)),
                            height=bar_ys, 
                            color=colors['undamaged'], 
                            label='damaged model',
                            alpha=0.8,
                            edgecolor='lightgrey',
                            #width=.3,
                            zorder=1
                            )
                    ### scatters
                    scatter_ys = [numpy.array(plot_results['undamaged'][key])-numpy.array(plot_results['damaged'][key]) for key in x_labels]
                    ax.scatter(
                               [x_i+(random.choice(range(-5,5))*0.05) for x_i, x in enumerate(x_labels) for val in scatter_ys[x_i]],
                               scatter_ys,
                               color=scatters['undamaged'],
                               edgecolors='white',
                               zorder=2
                               )
                    ax.set_title(out_file.replace('_', ' '), fontsize=30, fontweight='bold')
                    ax.legend(fontsize=25)
                    #ax.set_ylim(ymin=.4, ymax=1.)
                    ax.set_xticks(range(len(x_labels)))
                    ax.set_xticklabels(
                                       labels=[l.replace('.mean', '').replace('_', ' ') for l in x_labels],
                                       rotation=45,
                                       fontsize=25,
                                       fontweight='bold'
                                       )
                    #ax.set_ylabel('Cross-validation pearson correlation', fontsize=20)
                    ax.set_ylabel('Damage effect ($\Delta$ pearson correlation evaluation)', fontsize=25, fontweight='bold')
                    pyplot.savefig(os.path.join(
                                          'results', 
                                          '{}.jpg'.format(out_file), 
                                          ),
                                          dpi=300
                                          )
