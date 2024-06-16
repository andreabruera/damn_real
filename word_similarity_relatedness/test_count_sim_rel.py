import argparse
import fasttext
import gensim
import numpy
import os
import pickle
import random
import scipy
import sys

from gensim import downloader
from scipy import spatial, stats
from tqdm import tqdm

sys.path.append('..')

from utf_utils import transform_german_word
from utils import build_ppmi_vecs, read_lancaster_ratings

def compute_corr(dataset, dataset_name, present_words):
    missing_words = set()
    test_sims = list()
    for ws, val in dataset.items():
        marker = True
        w_ones = list()
        w_twos = list()
        if lang == 'de':
            for w in transform_german_word(ws[0]):
                try:
                    present_words.index(w)
                except ValueError:
                    continue
                w_ones.append(w)
            for w in transform_german_word(ws[1]):
                try:
                    present_words.index(w)
                except ValueError:
                    continue
                w_twos.append(w)
            if len(w_ones)<1:
                missing_words.add(ws[0])
            if len(w_twos)<1:
                missing_words.add(ws[1])
            if len(w_ones)<1 or len(w_twos)<1:
                marker = False
        else:
            for w in [ws[0], ws[0].capitalize()]:
                try:
                    present_words.index(w)
                except ValueError:
                    continue
                w_ones.append(w)
            for w in [ws[1], ws[1].capitalize()]:
                try:
                    present_words.index(w)
                except ValueError:
                    continue
                w_twos.append(w)
            #w_ones = [w for w in [ws[0], ws[0].capitalize()] if w in vocab]
            #w_twos = [w for w in [ws[1], ws[1].capitalize()] if w in vocab]
            if len(w_ones)<1:
                missing_words.add(ws[0])
            if len(w_twos)<1:
                missing_words.add(ws[1])
            if len(w_ones)<1 or len(w_twos)<1:
                #print(ws)
                marker = False
        if marker:
            #test_sims[ws] = val
            test_sims.append((w_ones, w_twos, val))
    if len(test_sims) == 0:
        #continue
        corr = None
        return corr
    if len(missing_words) > 0:
        print('missing words: {}'.format(missing_words))
    real = list()
    pred = list()
    #for k, v in test_sims.items():
    for w_ones, w_twos, v in test_sims:
        if 'tms' in dataset_name:
            ### for reaction times, it's the opposite
            real.append(1 - v)
        else:
            real.append(v)
        ### all possible transformations...
        current_pred = list()
        for w in w_ones:
            for w_two in w_twos:
                partial_pred = 1 - scipy.spatial.distance.cosine(model[w], model[w_two])
                current_pred.append(partial_pred)
        current_pred = numpy.average(current_pred)
        pred.append(current_pred)
    try:
        #corr = scipy.stats.pearsonr(real, pred).statistic
        corr = scipy.stats.spearmanr(real, pred).statistic
    except ValueError:
        corr = None
    return corr

def test_model(lang, case, model, vocab, results, datasets):
    if lang not in results.keys():
        results[lang] = dict()
    print('\n{}\n'.format(lang))
    #for case, model in lang_models.items():
    #    if case not in results[lang].keys():
    #        results[lang][case] = dict()
    #    vocab = vocabs[lang][case]
    present_words = list()
    for w in rows[lang]:
        try:
            vocab.index(w)
        except ValueError:
            continue
        present_words.append(w)
    for dataset_name, dataset in datasets[lang].items():
        if len(dataset) == 0:
            continue
        if case not in results[lang].keys():
            results[lang][case] = dict()
        if 'tms' not in dataset_name:
            #all_dataset = [(k, v) for k, v in dataset.items()]
            corr = compute_corr(dataset, dataset_name, present_words)
            if corr == None:
                print('error with {}'.format([lang, case, dataset_name]))
                continue
        else:
            corrs = list()
            for s, s_data in dataset.items():
                corr = compute_corr(s_data, dataset_name, present_words)
                if corr == None:
                    print('error with {} - subject {}'.format([lang, case, dataset_name, s]))
                else:
                    corrs.append(corr)
            corr = numpy.nanmean(corrs)
        print('\n')
        print('{} model'.format(case))
        print('correlation with {} dataset:'.format(dataset_name))
        print(corr)
        results[lang][case][dataset_name] = corr
    return results

def read_italian_cereb_tms(lang):
    sims = dict()
    test_vocab = set()
    lines = list()
    with open(os.path.join('..', 'tms', 'data', 'italian_tms_cereb.tsv')) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = [w for w in line]
                continue
            lines.append(line)
    conds = set([l[header.index('condition')] for l in lines])
    full_sims = dict()
    for name in conds:
        if lang == 'it':
            current_cond = [l for l in lines if l[header.index('condition')]==name]
            log_rts = [numpy.log10(float(l[header.index('RTs')].replace(',', '.'))) for l in current_cond]
            subjects = [int(l[header.index('Subject')]) for l in current_cond]
            w_ones = [l[header.index('noun')].lower() for l in current_cond]
            w_twos = [l[header.index('adj')].lower() for l in current_cond]
            vocab_w_ones = [w for ws in w_ones for w in [ws, ws.capitalize()]]
            test_vocab = test_vocab.union(set(vocab_w_ones))
            vocab_w_twos = [w for ws in w_twos for w in [ws, ws.capitalize()]]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
        else:
            full_sims[name] = list()
    if lang == 'it':
        full_sims = reorganize_tms_sims(sims)
    return full_sims, test_vocab

def read_german_ifg_tms(lang):
    sims = dict()
    test_vocab = set()
    lines = list()
    with open(os.path.join('..', 'tms', 'data', 'german_ifg_r.tsv')) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = [w for w in line]
                continue
            lines.append(line)
    ###
    conditions = {
                 (0.5,0.5) : 'aIFG',
                 (1.5, 1.5) : 'pIFG',
                 (1., 1.) : 'vertex',
                 (0.5, 1., 1.5) : 'all',
                 }
    full_sims = dict()
    for c, name in conditions.items():
        if lang == 'de':
            current_cond = [l for l in lines if float(l[header.index('condition')]) in c]
            subjects = [int(l[header.index('sub')]) for l in current_cond]
            log_rts = [float(l[header.index('log_rt')]) for l in current_cond]
            vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('word')])]
            test_vocab = test_vocab.union(set(vocab_w_ones))
            vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('category')])]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            w_ones = [l[header.index('word')] for l in current_cond]
            w_twos = [l[header.index('category')] for l in current_cond]
            sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
        else:
            full_sims[name] = list()
    if lang == 'de':
        full_sims = reorganize_tms_sims(sims)
    return full_sims, test_vocab

def reorganize_tms_sims(sims):
    full_sims = dict()
    for n, n_data in sims.items():
        full_sims[n] = dict()
        for s, ws, rt in n_data:
            if s not in full_sims[n].keys():
                full_sims[n][s] = dict()
            full_sims[n][s][ws] = rt
    return full_sims

def read_fern(lang):
    sims = {1 : dict(), 2 : dict()}
    test_vocab = set()
    if lang != 'en':
        pass
    else:
        for dataset in sims.keys():
            sims[dataset] = dict()
            file_path = os.path.join('data', 'fern{}.tsv'.format(dataset))
            with open(file_path) as i:
                for l_i, l in enumerate(i):
                    if l_i == 0:
                        continue
                    line = l.strip().split('\t')
                    test_vocab = test_vocab.union(set([line[0], line[1]]))
                    sims[dataset][(line[0], line[1])] = numpy.average(numpy.array(line[2:], dtype=numpy.float32))
    return sims[1], sims[2], test_vocab

def read_simlex(lang):
    if lang == 'de':
        file_path = os.path.join('data', 'MSimLex999_German.txt')
        indices = [0, 1, -1]
        sep = ','
    if lang == 'it':
        file_path = os.path.join('data', 'MSimLex999_Italian.txt')
        indices = [0, 1, -1]
        sep = ','
    if lang == 'en':
        file_path = os.path.join('data', 'SimLex-999.txt')
        indices = [0, 1, 3]
        sep = '\t'
    assert os.path.exists(file_path)
    sims = dict()
    test_vocab = set()
    with open(file_path) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split(sep)
            key = tuple(sorted([line[indices[0]], line[indices[1]]]))
            norm_key = set([m for k in key for w in k for m in transform_german_word(k)])
            test_vocab = test_vocab.union(norm_key)
            val = float(line[indices[2]].replace(',', '.'))
            sims[key] = val
    return sims, test_vocab

def read_ws353(lang):
    if lang == 'de':
        file_path = os.path.join('data', 'MWS353_German.txt')
    if lang == 'it':
        file_path = os.path.join('data', 'MWS353_Italian.txt')
    if lang == 'en':
        file_path = os.path.join('data', 'MWS353_English.txt')
    indices = [0, 1, -1]
    sep = ','
    assert os.path.exists(file_path)
    sims = dict()
    test_vocab = set()
    with open(file_path) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split(sep)
            key = tuple(sorted([line[indices[0]], line[indices[1]]]))
            norm_key = set([m for k in key for w in k for m in transform_german_word(k)])
            test_vocab = test_vocab.union(norm_key)
            val = float(line[indices[2]].replace(',', '.'))
            sims[key] = val
    return sims, test_vocab

def read_men(lang):
    sims = dict()
    test_vocab = set()
    if lang != 'en':
        print('the MEN dataset is not available for this language!')
    else:
        file_path = os.path.join('data', 'MEN_dataset_natural_form_full')
        assert os.path.exists(file_path)
        with open(file_path) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split()
                key = tuple(sorted([line[0], line[1]]))
                norm_key = set([m for k in key for w in k for m in transform_german_word(k)])
                test_vocab = test_vocab.union(norm_key)
                val = float(line[2].replace(',', '.'))
                sims[key] = val
    return sims, test_vocab

def read_mitchell(lang):
    assert lang == 'en'
    dimensions = list()
    with open(os.path.join('..', 'data', 'dimensions.tsv')) as i:
        for l in i:
            line = l.strip().split()
            assert len(line) >= 2
            dimensions.append(line)
    assert len(dimensions) == 25

    return dimensions

languages = [
             #'en',
             'de',
             'it',
             ]

rows = dict()
datasets = dict()
for lang in languages:
    men, men_vocab = read_men(lang)
    simlex, simlex_vocab = read_simlex(lang)
    ws353, ws353_vocab = read_ws353(lang)
    fern_one, fern_two, fern_vocab = read_fern(lang)
    germ_tms_ifg, germ_tms_ifg_vocab = read_german_ifg_tms(lang)
    ita_tms_cereb, ita_tms_cereb_vocab = read_italian_cereb_tms(lang)
    basic_vocab = men_vocab.union(
                                  simlex_vocab,
                                  ws353_vocab,
                                  fern_vocab,
                                  germ_tms_ifg_vocab,
                                  ita_tms_cereb_vocab,
                                  )
    rows[lang] = sorted(basic_vocab)
    datasets[lang] = dict()
    for dataset_name, dataset in [
                    ### similarity/relatedness
                    #('simlex999-sim', simlex),
                    #('ws353', ws353),
                    #('men', men), 
                    ### semantic network brain RSA
                    #('fern1', fern_one),
                    #('fern2', fern_two),
                    ## german TMS
                    ('de tms vertex', germ_tms_ifg['vertex']),
                    ('de tms pIFG', germ_tms_ifg['pIFG']),
                    ('de tms aIFG', germ_tms_ifg['aIFG']),
                    ## italian TMS
                    ('it tms cereb', ita_tms_cereb['cedx']),
                    ('it tms vertex', ita_tms_cereb['cz']),
                    ]:
        datasets[lang][dataset_name] = dataset

results = dict()
results_file = 'evaluation.tsv'
if os.path.exists(results_file):
    with open(results_file) as i:
        for l in i:
            line = l.strip().split('\t')
            assert len(line) == 4
            if line[0] not in results.keys():
                results[line[0]] = dict()
            if line[1] not in results[line[0]].keys():
                results[line[0]][line[1]] = dict()
            results[line[0]][line[1]][line[2]] = float(line[3])

senses = ['auditory', 'gustatory', 'haptic', 'olfactory', 'visual', 'hand_arm']   
lancaster_ratings = read_lancaster_ratings()

models = dict()
vocabs = dict()
for lang in languages:
    if lang not in results.keys():
        results[lang] = dict()
    models[lang] = dict()
    vocabs[lang] = dict()
    print('\n{}\n'.format(lang))
    for case in [
                 'fasttext',
                 'fasttext_aligned',
                 ]:
        print(case)
        #if case in results[lang].keys():
        #    continue
        if case == 'fasttext':
            model = fasttext.load_model('../../../dataset/word_vectors/{}/cc.{}.300.bin'.format(lang, lang))
            vocab = model.words
        elif case == 'fasttext_aligned':
            with open(os.path.join('..', 'pickles', 
                                   'ft_{}_aligned.pkl'.format(lang)
                                   ), 'rb') as i:
                model = pickle.load(i)
            vocab = model.keys()
        #models[lang][case] = {w : model[w] for w in vocab}
        #vocabs[lang][case] = [w for w in vocab]
        model = {w : model[w] for w in vocab}
        vocab = [w for w in vocab]
        results = test_model(lang, case, model, vocab, results, datasets)
    for corpus in [
               #'bnc',
               'wac',
               'tagged_wiki',
               'opensubs',
               'joint',
               'cc100',
               ]:
        print(corpus)
        if lang == 'en':
            if corpus == 'cc100':
                min_count = 500
            else:
                min_count = 100
        else:
            if corpus == 'cc100':
                min_count = 100
            else:
                min_count = 10
        with open(os.path.join(
                               '..', '..', 'psychorpus',
                               'pickles', lang, corpus, 
                               '{}_{}_uncased_vocab_min_{}.pkl'.format(lang, corpus, min_count),
                               ), 'rb') as i:
            vocab = pickle.load(i)
        with open(os.path.join(
                               '..', '..', 'psychorpus',
                               'pickles', lang, corpus, 
                               '{}_{}_uncased_word_freqs.pkl'.format(lang, corpus),
                               ), 'rb') as i:
            freqs = pickle.load(i)
        print('total size of the corpus: {:,} tokens'.format(sum(freqs.values())))
        with open(os.path.join(
                               '..', '..', 'psychorpus',
                               'pickles', lang, corpus, 
                               '{}_{}_coocs_uncased_min_{}_win_4.pkl'.format(lang, corpus, min_count),
                               ), 'rb') as i:
            coocs = pickle.load(i)
        ### selecting dimensions from ratings
        row_words = [w for w in rows[lang] if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0]
        if lang == 'en':
            ### mitchell
            for row_mode in ['', 'rowincol']:
                key = 'ppmi_{}_mitchell_{}_words'.format(corpus, row_mode)
                if row_mode == 'rowincol':
                    ctx_words = set([w for ws in read_mitchell(lang) for w in ws] + row_words)
                else:
                    ctx_words = [w for ws in read_mitchell(lang) for w in ws]
                trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, row_words, ctx_words, smoothing=False)
                #models[lang][key] = {k : v for k, v in trans_pmi_vecs.items()}
                model = {k : v for k, v in trans_pmi_vecs.items()}
                #vocabs[lang][key] = [w for w in trans_pmi_vecs.keys()]
                curr_vocab = [w for w in trans_pmi_vecs.keys()]
                results = test_model(lang, key, model, curr_vocab, results, datasets)
            ### lancaster
            filt_ratings = {w : freqs[w] for w in lancaster_ratings.keys() if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0}
            sorted_ratings = [w[0] for w in sorted(filt_ratings.items(), key=lambda item: item[1], reverse=True)]
            filt_perc = {w : v['minkowski3'] for w, v in lancaster_ratings.items() if w in vocab.keys() and w in freqs.keys() and vocab[w]!=0}
            sorted_perc = [w[0] for w in sorted(filt_perc.items(), key=lambda item: item[1], reverse=True)]
            #inv_sorted_ratings = [w[0] for w in sorted(filt_ratings.items(), key=lambda item: item[1])]
            #for percentage in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            #    freq = int(len(sorted_ratings)*percentage)
            #for freq in [7500, 10000, 12500, 15000, 17500]:
            for freq in tqdm([
                              100, 
                              200, 
                              500, 
                              750,
                              1000, 
                              2500, 5000, 7500,
                              10000, 12500, 15000, 17500,
                              20000, 25000
                              ]):
                for row_mode in ['', 'rowincol']:
                    for selection_mode in [
                                           'top', 
                                           'random', 
                                           'hi-perceptual', 
                                           'lo-perceptual',
                                           ]: 
                        key = 'ppmi_{}_lancaster_freq_{}_{}_{}_words'.format(corpus, selection_mode, row_mode, freq)
                        #if key in results[lang].keys():
                        #    continue
                        #ctx_words = [w for w in inv_sorted_ratings
                        #trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, ctx_words, ctx_words, smoothing=False)
                        ### using the basic required vocab for all tests as a basis set of words
                        #ctx_words = [w for w in sorted_ratings[:freq]]
                        if selection_mode == 'top':
                            if row_mode == 'rowincol':
                                ctx_words = set([w for w in sorted_ratings[:freq]]+row_words)
                            else:
                                ctx_words = [w for w in sorted_ratings[:freq]]
                        elif selection_mode == 'hi-perceptual':
                            if row_mode == 'rowincol':
                                ctx_words = set([w for w in sorted_perc[:freq]]+row_words)
                            else:
                                ctx_words = [w for w in sorted_perc[:freq]]
                        elif selection_mode == 'lo-perceptual':
                            if row_mode == 'rowincol':
                                ctx_words = set([w for w in sorted_perc[-freq:]]+row_words)
                            else:
                                ctx_words = [w for w in sorted_perc[-freq:]]
                        else:
                            random.seed(12)
                            idxs = random.sample(range(len(sorted_ratings)), k=freq)
                            if row_mode == 'rowincol':
                                ctx_words = set([sorted_ratings[i] for i in idxs]+row_words)
                            else:
                                ctx_words = [sorted_ratings[i] for i in idxs]
                        ctx_idxs = [vocab[w] for w in ctx_words]
                        trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, row_words, ctx_words, smoothing=False)
                        #models[lang][key] = {k : v for k, v in trans_pmi_vecs.items()}
                        model = {k : v for k, v in trans_pmi_vecs.items()}
                        #vocabs[lang][key] = [w for w in trans_pmi_vecs.keys()]
                        curr_vocab = [w for w in trans_pmi_vecs.keys()]
                        results = test_model(lang, key, model, curr_vocab, results, datasets)
        ### top-n frequencies
        filt_freqs = {w : f for w, f in freqs.items() if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0}
        sorted_freqs = [w[0] for w in sorted(filt_freqs.items(), key=lambda item: item[1], reverse=True)]
        for freq in tqdm([
                          100, 200, 500, 750,
                          1000, 
                          2500, 5000, 7500,
                          10000, 12500, 15000, 17500,
                          20000, 25000
                          ]):
            for row_mode in [
                             '', 
                             #'rowincol',
                             ]:
                for selection_mode in ['top', 'random']: 
                    #ctx_words = [w for w in inv_sorted_ratings
                    #trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, ctx_words, ctx_words, smoothing=False)
                    ### using the basic required vocab for all tests as a basis set of words
                    #ctx_words = [w for w in sorted_ratings[:freq]]
                    key = 'ppmi_{}_abs_freq_{}_{}_{}_words'.format(corpus, selection_mode, row_mode, freq)
                    #if key in results[lang].keys():
                    #    continue
                    if selection_mode == 'top':
                        if row_mode == 'rowincol':
                            ctx_words = set([w for w in sorted_freqs[:freq]]+row_words)
                        else:
                            ctx_words = [w for w in sorted_freqs[:freq]]
                    else:
                        random.seed(12)
                        idxs = random.sample(range(len(sorted_freqs)), k=freq)
                        if row_mode == 'rowincol':
                            ctx_words = set([sorted_freqs[i] for i in idxs]+row_words)
                        else:
                            ctx_words = [sorted_freqs[i] for i in idxs]
                    ### using the basic required vocab for all tests as a basis set of words
                    ctx_idxs = [vocab[w] for w in ctx_words]
                    trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, row_words, ctx_words, smoothing=False)
                    #models[lang][key] = {k : v for k, v in trans_pmi_vecs.items()}
                    #vocabs[lang][key] = [w for w in trans_pmi_vecs.keys()]
                    model = {k : v for k, v in trans_pmi_vecs.items()}
                    curr_vocab = [w for w in trans_pmi_vecs.keys()]
                    results = test_model(lang, key, model, curr_vocab, results, datasets)
    '''
    pruned_ratings = {w : dct for w, dct in ratings.items() if w in freqs.keys() and vocab[w]!=0}
    percent = int(len(pruned_ratings.items())*0.001)
    #percent = int(len(pruned_ratings.items())*0.05)
    #percent = int(len(pruned_ratings.items())*0.1)
    ### context words
    ### things improve when including the words directly
    ctx_words = set(pruned_test_words)
    #ctx_words = set()
    sem_dims = set([var for k,v in pruned_ratings.items() for var in v.keys()])
    for dim in sem_dims:
        if dim == 'concreteness':
            continue
        sorted_ws = sorted([(w, v[dim]) for w, v in pruned_ratings.items()], key=lambda item: item[1])
        ctx_words = ctx_words.union(set([w for w, val in sorted_ws[-percent:]]))
        ### also adding super abstract words
        #    ctx_words = ctx_words.union(set([w[0] for w in sorted_ws[:one_percent]]))
        #    ctx_words = ctx_words.union(set([w for w, val in sorted_ws[-one_percent:]]))
    print('considering {} context words'.format(len(ctx_words)))
    ctx_words = sorted(ctx_words)
    ctx_idxs = [vocab[w] for w in ctx_words]
    vecs = {w : numpy.array([coocs[vocab[w]][idx] if idx in coocs[vocab[w]].keys() else 0 for idx in ctx_idxs]) for w in pruned_test_words}
    ### pmi
    ### building the PPMI matrix
    ### things are better when including in the rows the words from MEN...
    trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, ctx_words, ctx_words, smoothing=True)
    '''

### writing to file
with open(results_file, 'w') as o:
    for lang, lang_data in results.items():
        for case, case_data in lang_data.items():
            for dataset, corr in case_data.items():
                o.write('{}\t{}\t{}\t{}\n'.format(lang, case, dataset, corr))
