import argparse
import fasttext
#import gensim
import numpy
import os
import pickle
import random
import scipy
import sklearn

from scipy import spatial
from sklearn import linear_model
from tqdm import tqdm

from fmri_loaders import read_abstract_ipc, read_fern, read_fern_categories
from meeg_loaders import read_dirani_n400
from behav_loaders import read_italian_behav, read_german_behav
from tms_loaders import read_it_distr_learn_tms, read_de_sound_act_tms, read_de_sem_phon_tms
from simrel_norms_loaders import read_men, read_simlex, read_ws353
from utf_utils import transform_german_word

def check_dataset_words(args, dataset_name, dataset, present_words, trans_from_en, ):
    #print('checking if words appear in the dictionary...')
    missing_words = set()
    test_sims = list()
    if type(dataset) != list:
        dataset = [(k, v) for k, v in dataset.items()]
    for ws, val in dataset:
        ### prototypes
        if type(ws[0]) != tuple:
            first_word = set([ws[0]])
        else:
            for w in ws[0]:
                assert type(w) == str
            first_word = ws[0]
        marker = True
        w_ones = list()
        w_twos = list()
        if 'fern' in dataset_name and args.lang in ['de', 'it']:
            ### word one
            for word in first_word:
                try:
                    candidates = trans_from_en[args.lang][word]
                    for c in candidates:
                        try:
                            present_words.index(c)
                            w_ones.append(c)
                        except ValueError:
                            #print(c)
                            pass
                except KeyError:
                    #print(ws[0])
                    pass
            ### word two 
            try:
                candidates = trans_from_en[args.lang][ws[1]]
                for c in candidates:
                    try:
                        present_words.index(c)
                        w_twos.append(c)
                    except ValueError:
                        #print(c)
                        pass
            except KeyError:
                #print(ws[1])
                pass
        else:
            if args.lang == 'de':
                for word in first_word:
                    #print(word)
                    for w in transform_german_word(word):
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
            else:
                for word in first_word:
                    for w in [word.lower(), word.capitalize()]:
                        try:
                            present_words.index(w)
                        except ValueError:
                            continue
                        w_ones.append(w)
                for w in [ws[1].lower(), ws[1].capitalize()]:
                    try:
                        present_words.index(w)
                    except ValueError:
                        continue
                    w_twos.append(w)
        if len(w_ones)<1:
            missing_words = missing_words.union(first_word)
        if len(w_twos)<1:
            missing_words.add(ws[1])
        if len(w_ones)<1 or len(w_twos)<1:
            marker = False
        if marker:
            test_sims.append((w_ones, w_twos, val))
    return test_sims, missing_words

def compute_corr(args, model, dataset, dataset_name, present_words, trans_from_en,):
    test_sims, missing_words = check_dataset_words(args, dataset_name, dataset, present_words, trans_from_en,)
    #if len(prototypes.keys()) > 0:
    #    proto_vecs = prototypes
    #if len(test_sims) == 0:
    #    corr = None
    #    return corr
    assert len(test_sims) > 0
    real = list()
    pred = list()
    for w_ones, w_twos, v in test_sims:
        real.append(v)
        ### all possible transformations...
        '''
        if len(prototypes.keys()) > 0:
            current_pred = list()
            proto_mode = dataset_name.split('#')[-1]
            proto_modes = [
                         'all', 
                         'both_pos',
                         'both_pos-topten',
                         'both_pos-topfifty',
                         'both_neg',
                         'matched_excl',
                         'matched_excl-topten',
                         'matched_excl-topfifty',
                         'matched_non_excl',
                         'matched_non_excl-topten',
                         'matched_non_excl-topfifty',
                         'opposite_excl',
                         'opposite_excl-topten',
                         'opposite_excl-topfifty',
                         'opposite_non_excl',
                         'opposite_non_excl-topten',
                         'opposite_non_excl-topfifty',
                         ]
            assert proto_mode in proto_modes
            if 'top' in proto_mode:
                val = proto_mode.split('-')[-1]
            #print(proto_mode)
            for w_two in w_twos:
                if proto_mode in ['all', 'both_pos', 'both_pos-topten', 'both_pos-topfifty', 'both_neg']:
                    #proto_mode = proto_mode.replace('-', '_')
                    vec_one = proto_vecs['{}'.format(proto_mode)]
                else:
                    if 'non-excl' in proto_mode:
                        if 'matched' in proto_mode:
                            ### sound
                            if 'er' in w_ones[0]:
                                if 'top' in proto_mode:
                                    vec_one = proto_vecs['sound_pos-{}'.format(val)]
                                else:
                                    vec_one = proto_vecs['sound_pos']
                            elif 'andlung' in w_ones[0]:
                                if 'top' in proto_mode:
                                    vec_one = proto_vecs['action_pos-{}'.format(val)]
                                else:
                                    vec_one = proto_vecs['action_pos']
                            else:
                                raise RuntimeError()
                        elif 'opposite' in proto_mode:
                            ### sound
                            if 'er' in w_ones[0]:
                                vec_one = proto_vecs['sound_neg']
                            elif 'andlung' in w_ones[0]:
                                vec_one = proto_vecs['action_neg']
                            else:
                                raise RuntimeError()
                        else:
                            raise RuntimeError()
                    elif 'excl' in proto_mode:
                        if 'matched' in proto_mode:
                            ### sound
                            if 'er' in w_ones[0]:
                                if 'top' in proto_mode:
                                    vec_one = proto_vecs['sound_pos_action_neg-{}'.format(val)]
                                else:
                                    vec_one = proto_vecs['sound_pos_action_neg']
                            elif 'andlung' in w_ones[0]:
                                if 'top' in proto_mode:
                                    vec_one = proto_vecs['action_pos_sound_neg-{}'.format(val)]
                                else:
                                    vec_one = proto_vecs['action_pos_sound_neg']
                            else:
                                raise RuntimeError()
                        elif 'opposite' in proto_mode:
                            ### sound
                            if 'er' in w_ones[0]:
                                vec_one = proto_vecs['action_pos_sound_neg']
                            elif 'andlung' in w_ones[0]:
                                vec_one = proto_vecs['sound_pos_action_neg']
                            else:
                                raise RuntimeError()
                        else:
                            raise RuntimeError()
                    else:
                        raise RuntimeError()
                ### always using dissimilarity
                partial_pred = scipy.spatial.distance.cosine(vec_one, model[w_two])
                #partial_pred = 1 - scipy.stats.spearmanr(vec_one, model[w_two]).statistic
                current_pred.append(partial_pred)
            current_pred = numpy.average(current_pred)
        else:
        '''
        current_pred = list()
        for w in w_ones:
            for w_two in w_twos:
                ### always using dissimilarity
                partial_pred = scipy.spatial.distance.cosine(model[w], model[w_two])
                #partial_pred = 1 - scipy.stats.spearmanr(model[w], model[w_two]).statistic
                current_pred.append(partial_pred)
        current_pred = numpy.average(current_pred)
        pred.append(current_pred)
    corr = scipy.stats.spearmanr(real, pred).statistic
    '''
    try:
        #corr = scipy.stats.pearsonr(real, pred).statistic
        corr = scipy.stats.spearmanr(real, pred).statistic
    except ValueError:
        corr = None
    '''
    return corr, missing_words

def write_res(args, case, dataset_name, corr):
    corpus_fold = case.split('_')[1] if 'ppmi' in case else case
    details = '_'.join(case.split('_')[2:]) if 'ppmi' in case else case
    out_folder = os.path.join(
                              'test_results',
                              args.lang, 
                              corpus_fold, 
                              details,
                              )
    os.makedirs(out_folder, exist_ok=True)
    out_f = os.path.join(out_folder, '{}.tsv'.format(dataset_name))
    with open(out_f, 'w') as o:
        o.write('{}\t{}\t{}\t'.format(args.lang, case, dataset_name))
        for c in corr:
            o.write('{}\t'.format(c))
    print(out_f)

def check_present_words(args, rows, vocab):
    present_words = list()
    for w in rows:
        ### for fasttext in german we only use uppercase!
        if w[0].isupper() == False and args.model=='fasttext' and args.lang=='de':
            continue
        try:
            vocab.index(w)
        except ValueError:
            continue
        present_words.append(w)
    return present_words

def test_model(args, case, model, vocab, datasets, present_words, trans_from_en):
    if args.stat_approach != 'simple':
        datasets = bootstrapper(args, datasets, )
    else:
        datasets = {k : [v] for k, v in datasets.items()}
    for dataset_name, dataset in datasets.items():
        assert type(dataset) == list
        assert len(dataset) in [1, 1000]
        corr = list()
        missing_words = set()
        ### bootstrapping/iterations should be hard-coded now...
        for iter_dataset in tqdm(dataset):
            iter_corrs = list()
            for s, s_data in iter_dataset.items():
                curr_corr, new_miss = compute_corr(args, model, s_data, dataset_name, present_words, trans_from_en)
                missing_words = missing_words.union(new_miss)
                if corr == None:
                    print('error with {}'.format([args.lang, case, dataset_name]))
                    continue
                iter_corrs.append(curr_corr)
            if args.stat_approach == 'simple':
                corr.extend(iter_corrs)
            else:
                iter_corr = numpy.average(iter_corrs)
                corr.append(iter_corr)

        print('\n')
        print('{} model'.format(case))
        print('correlation with {} dataset:'.format(dataset_name))
        print(numpy.nanmean(corr))
        if len(missing_words) > 0:
            print('missing words: {}'.format(missing_words))
        write_res(args, case, dataset_name, corr)
        #return results

def bootstrapper(args, full_data, ):
    ### bootstrapping with b=block_size of the original data
    ### Politis et al 1999, Bootstrapping page 198 "Indeed, for b too close to n 
    ### all subsample statistics (On,b,i or On,b,t) will be almost equal to On, 
    ### resulting in the subsampling distribution being too tight and in 
    ### undercoverage of subsampling confidence intervals [...]
    ###  On the other hand, if b is too small, the intervals can undercover or overcover depending on the state of nature. 
    ### This leaves a number of b-values in
    ### the "right range" where we would expect almost correct results, at least
    ### for large sample sizes. Hence, in this range, the confidence intervals should
    ### be "stable" when considered as a function of the block size. This idea is
    ### exploited by computing subsampling intervals for a large number of block
    ### sizes b and then looking for a region where the intervals do not change
    ### see figure in page 191
    ### we do not estimate it, but use values used in their simulation (page 208)
    if args.stat_approach == 'residualize':
        proportions = [
                       0.5
                       ]
    else:
        if 'behav' in args.dataset:
            proportions = [
                           4/256,
                           8/256,
                           16/256,
                           32/256,
                           64/256,
                           ]
        else:
            ### Riccardo De Bin, Silke Janitza, Willi Sauerbrei, Anne-Laure Boulesteix, 
            ### Subsampling Versus Bootstrapping in Resampling-Based Model Selection for 
            ### Multivariable Regression, Biometrics, Volume 72, Issue 1, 
            ### March 2016, Pages 272–280
            proportions = [
                    0.632
                    #0.5
                    ]
    ### labels
    labels = list(full_data.keys())
    ### subjects
    all_subjects = [list(v.keys()) for v in full_data.values()]
    # checking
    n_subjects = set([len(v) for v in all_subjects])
    assert len(n_subjects) == 1
    n_subjects = list(n_subjects)[0]
    subjects = list(set([val for v in all_subjects for val in v]))
    assert len(subjects) == n_subjects
    ### for tms we fix n=20
    tms_datasets = [
                   'it_distr-learn',
                   'de_sem-phon', 
                   'de_sound-act_'
                   ]
    if args.dataset not in tms_datasets:
        n_iter_sub = max(1, int(n_subjects*random.choice(proportions)))
    else:
        n_iter_sub = 20
    ### here we create 1000
    boot_data = {l : list() for l in labels}
    if args.stat_approach == 'residualize':
        print('residualizing...')
    for _ in range(1000):
        iter_subs = random.sample(subjects, k=n_iter_sub)
        ### for tms we fix n=20
        if args.dataset not in tms_datasets:
            iter_data_keys = {l : 
                               {s : random.sample(
                                                 full_data[l][s].keys(), 
                                                 k=int(len(full_data[l][s].keys())*random.choice(proportions))
                                                 #k=100,
                                                 ) for s in iter_subs}
                                                 for l in labels}
        else:
            iter_data_keys = {l : 
                               {s : random.sample(
                                                 full_data[l][s].keys(), 
                                                 k=20,
                                                 ) for s in iter_subs}
                                                 for l in labels}
        iter_data = {l : {s : {k : full_data[l][s][k] for k in iter_data_keys[l][s]} for s in iter_subs} for l in labels}
        ### residualization
        if args.stat_approach == 'residualize':
            struct_train_data = {l : {s : {k : full_data[l][s][k] for k in full_data[l][s].keys() if k not in iter_data_keys[l][s]} for s in iter_subs} for l in labels}
            flat_train_data = [(l, s, rt) for l, l_res in struct_train_data.items() for s, s_res in l_res.items() for k, rt in s_res.items()]
            flat_test_data = [(l, s, k, rt) for l, l_res in iter_data.items() for s, s_res in l_res.items() for k, rt in s_res.items()]
            model = sklearn.linear_model.LinearRegression()
            ### we remove tms/sham -> t[0] per subject -> t[1]
            model.fit(
                      [[labels.index(t[0]), t[1]] for t in flat_train_data],
                      [[t[2]] for t in flat_train_data],
                      )
            preds = model.predict(
                                  [[labels.index(t[0]),t[1]] for t in flat_test_data]
                                  )
            #print(preds)
            #print([t[3] for t in flat_test_data])
            residuals = [(real[0], real[1], real[2], real[3]-pred[0]) for real, pred in zip(flat_test_data, preds)]
            for l, s, k, r in residuals:
                iter_data[l][s][k] = r
        for l, l_data in iter_data.items():
            boot_data[l].append(l_data)
    return boot_data

def check_args(args):
    ### checking language if in first part of the name
    if '_' in args.dataset:
        assert args.dataset.split('_')[0] == args.lang
    if args.dataset == 'men' and args.lang != 'en':
        raise AssertionError

def load_dataset(args, trans_from_en):
    if args.dataset == 'en_men':
        data, vocab = read_men(args)
    if '999' in args.dataset:
        data, vocab = read_simlex(args)
    if '353' in args.dataset:
        data, vocab = read_ws353(args)
    if 'fern' in args.dataset:
        if 'all' in args.dataset:
            data, vocab = read_fern(args, trans_from_en)
        elif 'categories' in args.dataset:
            data, vocab = read_fern_categories(args, trans_from_en)
    if 'dirani' in args.dataset:
        data, vocab = read_dirani_n400(args)
    if 'abstract' in args.dataset:
        data, vocab = read_abstract_ipc(args)
    if 'de_behav' in args.dataset:
        data, vocab = read_german_behav(args)
    if 'it_behav' in args.dataset:
        data, vocab = read_italian_behav(args)
    if 'sem-phon' in args.dataset:
        data, vocab = read_de_sem_phon_tms(args)
    if 'sound-act' in args.dataset:
        data, vocab, prototypes = read_de_sound_act_tms(args)
    if 'distr-learn' in args.dataset:
        data, vocab = read_it_distr_learn_tms(args)
    return vocab, data

def load_static_model(args):
    print('loading {}'.format(args.model))
    base_folder = os.path.join(
                                '/',
                                'data',
                                'u_bruera_software',
                                #'tu_bruera',
                                'word_vectors', 
                                args.lang, 
                                )
    if args.model == 'fasttext':
        model = fasttext.load_model(
                                    os.path.join(
                                        base_folder,
                                        'cc.{}.300.bin'.format(args.lang)
                                        )
                                    )
        vocab = model.words
    elif case == 'conceptnet':
        with open(
                os.path.join(
                    base_folder,
                   'conceptnet_{}.pkl'.format(args.lang)
                   ), 'rb') as i:
            model = pickle.load(i)
        vocab = model.keys()
    elif case == 'fasttext_aligned':
        with open(
                  os.path.join(
                            base_folder,
                           'ft_{}_aligned.pkl'.format(args.lang)
                           ), 'rb') as i:
            model = pickle.load(i)
        vocab = model.keys()
    model = {w : model[w] for w in vocab}
    vocab = [w for w in vocab]

    return model, vocab

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        '--model',
                        choices=[
                                 'fasttext',
                                 'fasttext_aligned',
                                 'conceptnet',
                                 'bnc',
                                 'wac',
                                 'tagged_wiki',
                                 'opensubs',
                                 'joint',
                                 'cc100',
                                 ],
                        required=True,
                        )
    parser.add_argument(
                        '--lang',
                        choices=[
                                 'en',
                                 'de',
                                 'it',
                                 ],
                        required=True
                        )
    parser.add_argument(
                        '--dataset',
                        choices=[
                                ### sim-lex norms
                                'simlex999',
                                'ws353',
                                'en_men',
                                ### fmri
                                'fern2-all',
                                'fern1-categories',
                                'fern1-all',
                                'fern2-categories',
                                'de_abstract-fmri',
                                ### meeg
                                'dirani-n400',
                                ### behav
                                'de_behav',
                                'it_behav',
                                ### tms
                                'de_sem-phon',
                                'de_sound-act',
                                'it_distr-learn',
                                ],
                        required=True,
                        )
    parser.add_argument(
                        '--stat_approach',
                        choices=['simple', 'bootstrap', 'residualize'],
                        )
    #senses = ['auditory', 'gustatory', 'haptic', 'olfactory', 'visual', 'hand_arm']   
    args = parser.parse_args()
    check_args(args)

    return args
