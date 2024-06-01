import argparse
import fasttext
import gensim
import numpy
import os
import pickle
import random
import scipy

import gensim.downloader as api

from scipy import spatial, stats
from tqdm import tqdm

from utils import bins_rsa_test, full_dataset_rsa_test, build_ppmi_vecs, read_fernandino_ratings, read_ratings, read_fernandino

ratings = read_ratings(hand=True)

corpus = [
           #'opensubs',
           'joint',
           ][0]
min_count = [
              10, 
              #100,
              ][0]
percent_val = [
                1,
                ][0]
win_size = [
             4, 
             ][0]

for lang in [
                #'en',
                 'de',
                 ]:
    for smoothing, marker in [ 
                              (True,'{}_pmi_smooth75'.format(lang)), 
                              #(False, 'pmi_unsmoothed')
                              ]:
        full_corpus = '{}_coocs_uncased_min_{}_win_{}'.format(corpus, min_count, win_size)
        ### loading
        print('\n\n{} - min {} - {} percent - window size {}'.format(corpus, min_count, percent_val*100, win_size))
        with open(os.path.join(
                               'pickles', lang, corpus, 
                               '{}_{}_uncased_word_freqs.pkl'.format(lang, corpus),
                               ), 'rb') as i:
            freqs = pickle.load(i)
        with open(os.path.join(
                               'pickles', lang, corpus, 
                               '{}_{}_uncased_word_pos.pkl'.format(lang, corpus),
                               ), 'rb') as i:
            pos = pickle.load(i)
        print('total size of the corpus: {} tokens'.format(sum(freqs.values())))
        try:
            with open(os.path.join(
                                   'pickles', lang, corpus, 
                                   '{}_{}_coocs_uncased_min_{}_win_{}.pkl'.format(lang, corpus, min_count, win_size),
                                   ), 'rb') as i:
                coocs = pickle.load(i)
        except FileNotFoundError:
            print('this model is missing!')
            continue
        with open(os.path.join(
                               'pickles', lang, corpus, 
                               '{}_{}_uncased_vocab_min_{}.pkl'.format(lang, corpus, min_count),
                               ), 'rb') as i:
            vocab = pickle.load(i)
        trans = dict()
        if lang == 'de':
            ### reading translations
            inv_trans = dict()
            fs = [
                 'lancaster_english_to_german.tsv',
                 'fernandino_english_to_german.tsv', 
                 ]
            for f in fs:
                gers = dict()
                with open(os.path.join('data', f)) as i:
                    for l_i, l in enumerate(i):
                        if l_i == 0:
                            continue
                        line = l.strip().split('\t')
                        if f == fs[-1]:
                            try:
                                gers[line[1].strip()] += 1
                            except KeyError:
                                gers[line[1].strip()] = 1
                        trans[line[0].strip()] = line[1].strip().replace('-', '_').lower()
                        inv_trans[line[1].strip().replace('-', '_').lower()] = line[0].strip()
            for w, val in gers.items():
                if val > 1:
                    print(w)
        words, data = read_fernandino(vocab, pos, lang=lang, trans=trans)
        fern_ratings = read_fernandino_ratings()

        test_words = set(words[1]).union(set(words[2]))
        present=dict()
        print('missing words:')
        missing = list()
        for w in test_words:
            if lang == 'de':
                try:
                    w = trans[w].lower()
                except KeyError:
                    missing.append(w)
                    continue
            if w not in freqs.keys():
                missing.append(w)
                print('{} - not appearing at all'.format(w))
                continue
            if vocab[w]==0:
                missing.append(w)
                print([w, pos[w]])
                continue
            if vocab[w] not in coocs.keys():
                missing.append(w)
                print([w, freqs[w]])
                continue
            else:
                present[w] = freqs[w]
        print('\n')
        print('corpus: {}'.format(corpus))
        print('number of words missing: {}'.format(len(missing)))
        print(missing)
        min_n = min(list(present.values()))
        max_n = max(list(present.values()))
        lower_ten_ws = [w for w, val in present.items() if val < 10]
        lower_fifty_ws = [w for w, val in present.items() if val < 50]
        lower_hundred_ws = [w for w, val in present.items() if val < 100]
        ten_n = sum([1 for val in present.values() if val >= 10])
        fifty_n = sum([1 for val in present.values() if val >= 50])
        hundred_n = sum([1 for val in present.values() if val >= 100])
        avg_n = numpy.average(list(present.values()))
        med_n = numpy.median(list(present.values()))
        print('minimum number of mentions: {}'.format(min_n))
        print('maximum number of mentions: {}'.format(max_n))
        print('number of items above 10 mentions: {}'.format(ten_n))
        print('items with less than 10 mentions:')
        print(lower_ten_ws)
        print('number of items above 50 mentions: {}'.format(fifty_n))
        print('items with less than 50 mentions:')
        print(lower_fifty_ws)
        print('number of items above 100 mentions: {}'.format(hundred_n))
        print('items with less than 100 mentions:')
        print(lower_hundred_ws)
        print('average number of mentions: {}'.format(avg_n))
        print('median number of mentions: {}'.format(med_n))

        print('total number of ratings words available: {}'.format(len(ratings.keys())))
        ### removing rare words
        if lang == 'de':
            pruned_test_words = [w for w in present.keys()]
            pruned_ratings = {trans[w] : dct for w, dct in ratings.items() if w in trans.keys() and trans[w] in freqs.keys() and vocab[trans[w]]!=0 and vocab[trans[w]] in coocs.keys()}
        else:
            pruned_test_words = [w for w in test_words if w not in missing]
            #freq_missing_ratings = set(ratings.keys()).difference(set(freqs.keys()))
            #voc_missing_ratings = [w for w, dct in ratings.items() if w in freqs.keys() and vocab[w]==0]
            #freq_missing_ratings = [w for w, dct in ratings.items() if w not in freqs.keys()]
            pruned_ratings = {w : dct for w, dct in ratings.items() if w in freqs.keys() and vocab[w]!=0 and vocab[w] in coocs.keys()}
        print('number of ratings words actually used: {}'.format(len(pruned_ratings.keys())))
        #percent = int(len(pruned_ratings.items())*0.001)
        #percent = int(len(pruned_ratings.items())*0.06)
        percent = int(len(pruned_ratings.items())*percent_val)
        ### all words
        #percent = int(len(pruned_ratings.items()))
        ### context words
        ### things improve when including the words directly
        ctx_words = set(pruned_test_words)
        #ctx_words = set()
        sem_dims = set([var for k,v in pruned_ratings.items() for var in v.keys()])
        for dim in sem_dims:
            sorted_ws = sorted([(w, v[dim]) for w, v in pruned_ratings.items()], key=lambda item: item[1])
            ctx_words = ctx_words.union(set([w for w, val in sorted_ws[-percent:]]))
            if dim == 'concreteness':
            #    continue
                ### also adding super abstract words
                ctx_words = ctx_words.union(set([w[0] for w in sorted_ws[:percent]]))
            #    ctx_words = ctx_words.union(set([w for w, val in sorted_ws[-one_percent:]]))
        print('considering {} context words'.format(len(ctx_words)))
        ctx_words = sorted(ctx_words)
        ctx_idxs = [vocab[w] for w in ctx_words]
        vecs = {w : numpy.array([coocs[idx_one][idx_two] if idx_two in coocs[idx_one].keys() else 0 for idx_two in ctx_idxs]) for idx_one in ctx_idxs}
        ### pmi
        ### building the PPMI matrix
        ### things are better when including in the rows the words from MEN...
        if smoothing:
            trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, ctx_words, ctx_words, smoothing=True)
            if lang == 'de':
                trans_pmi75_vecs = {inv_trans[w] : v for w, v in trans_pmi_vecs.items()}
        else:
            trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, ctx_words, ctx_words, smoothing=False)
            if lang == 'de':
                trans_pmi_vecs = {inv_trans[w] : v for w, v in trans_pmi_vecs.items()}
        ### re-building the matrix
        mtrx = numpy.array([trans_pmi_vecs[w] for w in ctx_words])
        assert mtrx.shape == (len(ctx_words), len(ctx_words))
        out_f = os.path.join(
                             'damaged_pickles', lang, full_corpus, marker, 
                               )
        os.makedirs(out_f, exist_ok=True)
        with open(os.path.join(
                                out_f,
                               'words_rows_cols.pkl',
                               ), 'wb') as i:
            pickle.dump(ctx_words, i)
        with open(os.path.join(
                               out_f,
                               'undamaged_coocs.pkl',
                               ), 'wb') as i:
            pickle.dump(mtrx, i)
