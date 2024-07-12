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

def check_dataset_words(dataset_name, dataset, present_words, trans_from_en, prototyping=False):
    print('checking if words appear in the dictionary...')
    missing_words = set()
    test_sims = list()
    if prototyping:
        dataset = {(w, w) : '' for w in dataset} 
    with tqdm() as counter:
        for ws, val in dataset.items():
            marker = True
            w_ones = list()
            w_twos = list()
            if 'fern' in dataset_name and lang in ['de', 'it']:
                ### word one
                try:
                    candidates = trans_from_en['de'][ws[0]]
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
                    candidates = trans_from_en['de'][ws[1]]
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
            if len(w_ones)<1:
                missing_words.add(ws[0])
            if len(w_twos)<1:
                missing_words.add(ws[1])
            if len(w_ones)<1 or len(w_twos)<1:
                #print(ws)
                marker = False
            counter.update(1)
            if marker:
                #test_sims[ws] = val
                test_sims.append((w_ones, w_twos, val))
    if len(missing_words) > 0:
        print('missing words: {}'.format(missing_words))
    return test_sims

def compute_corr(dataset, dataset_name, present_words, prototypes, trans_from_en, printing=0):
    test_sims = check_dataset_words(dataset_name, dataset, present_words, trans_from_en)
    if len(prototypes.keys()) > 0:
        proto_sims = dict()
        for k, v in prototypes.items():
            proto_sims[k] = set([w for _ in check_dataset_words(v, present_words, prototyping=True) for __ in _[:2] for w in __])
        proto_vecs = dict()
        for k, v in proto_sims.items():
            if printing == 0:
                #print([k, v])
                pass
            assert len(v) > 0
            current_vecs = list()
            for w in v:
                current_vecs.append(model[w])
            current_vec = numpy.average(current_vecs, axis=0)
            proto_vecs[k] = current_vec

    if len(test_sims) == 0:
        #continue
        corr = None
        return corr
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
        if len(prototypes.keys()) > 0:
            current_pred = list()
            for proto_mode in [
                               'both', 
                               #'none',
                               ]: 
                for w_two in w_twos:
                    vec_one = proto_vecs[proto_mode]
                    '''
                    if 'er' in w_ones[0]:
                        #vec_one = proto_vecs['sound']
                        #print([w_ones, 'sound'])
                    elif 'andlung' in w_ones[0]:
                        vec_one = proto_vecs['none']
                        #vec_one = proto_vecs['action']
                        #print([w_ones, 'action'])
                    else:
                        raise RuntimeError()
                    '''
                    ### this is similarity!
                    partial_pred = 1 - scipy.spatial.distance.cosine(vec_one, model[w_two])
                    #partial_pred = scipy.stats.spearmanr(vec_one, model[w_two]).statistic
                    current_pred.append(partial_pred)
            current_pred = numpy.average(current_pred)
        else:
            current_pred = list()
            for w in w_ones:
                for w_two in w_twos:
                    partial_pred = 1 - scipy.spatial.distance.cosine(model[w], model[w_two])
                    #partial_pred = scipy.stats.spearmanr(model[w], model[w_two]).statistic
                    current_pred.append(partial_pred)
            current_pred = numpy.average(current_pred)
        pred.append(current_pred)
    try:
        #corr = scipy.stats.pearsonr(real, pred).statistic
        corr = scipy.stats.spearmanr(real, pred).statistic
    except ValueError:
        corr = None
    return corr

def test_model(lang, case, model, vocab, datasets, trans_from_en):
    #if lang not in results.keys():
    #    results[lang] = dict()
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
    for dataset_name, dataset_proto in datasets[lang].items():
        dataset = dataset_proto[0]
        prototypes = dataset_proto[1]
        if len(dataset) == 0:
            continue
        #if case not in results[lang].keys():
        #    results[lang][case] = dict()
        if 'tms' not in dataset_name:
            #all_dataset = [(k, v) for k, v in dataset.items()]
            corr = compute_corr(dataset, dataset_name, present_words, prototypes, trans_from_en)
            if corr == None:
                print('error with {}'.format([lang, case, dataset_name]))
                continue
            corr = [corr]
        else:
            corr = list()
            printing = 0
            for s, s_data in dataset.items():
                curr_corr = compute_corr(s_data, dataset_name, present_words, prototypes, trans_from_en, printing=printing)
                if curr_corr == None:
                    print('error with {} - subject {}'.format([lang, case, dataset_name, s]))
                else:
                    corr.append(curr_corr)
                printing += 1
            #corr = numpy.nanmean(corrs)
        print('\n')
        print('{} model'.format(case))
        print('correlation with {} dataset:'.format(dataset_name))
        print(numpy.nanmean(corr))
        corpus_fold = case.split('_')[1] if 'ppmi' in case else case
        details = '_'.join(case.split('_')[2:]) if 'ppmi' in case else case
        out_folder = os.path.join(
                                  'results',
                                  lang, 
                                  corpus_fold, 
                                  details,
                                  )
        os.makedirs(out_folder, exist_ok=True)
        with open(os.path.join(out_folder, '{}.tsv'.format(dataset_name)), 'w') as o:
            o.write('{}\t{}\t{}\t'.format(lang, case, dataset_name))
            for c in corr:
                o.write('{}\t'.format(c))
            o.write('\n')
        #results[lang][case][dataset_name] = corr
    #return results

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
            #log_rts = [float(l[header.index('RTs')].replace(',', '.')) for l in current_cond]
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

def read_german_pipl_tms(lang):
    sims = dict()
    test_vocab = set()
    lines = list()
    prototypes = {
                  'pos_action' : list(), 
                  'pos_sound' : list(),
                  'neg_action' : list(),
                  'neg_sound' : list(), 
                  'all' : list(),
                  'both' : list(),
                  'none' : list(),
                  'only_action' : list(), 
                  'only_sound' : list(),
                  }
    with open(os.path.join('..', 'tms', 'data', 'de_tms_pipl.tsv')) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = [w for w in line]
                continue
            if line[header.index('action_word')] == '1' and line[header.index('sound_word')] == '-1':
                prototypes['only_action'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '-1' and line[header.index('sound_word')] == '1':
                prototypes['only_sound'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '1' and line[header.index('sound_word')] == '1':
                prototypes['both'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '-1' and line[header.index('sound_word')] == '-1':
                prototypes['none'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '1':
                prototypes['pos_action'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '-1':
                prototypes['neg_action'].append(line[header.index('stimulus')])
            if line[header.index('sound_word')] == '1':
                prototypes['pos_sound'].append(line[header.index('stimulus')])
            if line[header.index('sound_word')] == '-1':
                prototypes['neg_sound'].append(line[header.index('stimulus')])
            #if line[header.index('action_word')] == '-1' and line[header.index('sound_word')] == '1':
            #    prototypes['action'].append(line[header.index('stimulus')])
            #if line[header.index('sound_word')] == '-1' and line[header.index('action_word')] == '1':
            #    prototypes['sound'].append(line[header.index('stimulus')])
            if 'lexical_decision' in line:
                continue
            prototypes['all'].append(line[header.index('stimulus')])
            lines.append(line)
    full_sims = dict()
    conditions = set([l[header.index('condition')] for l in lines])
    tasks = set([l[header.index('task')] for l in lines])
    for c in conditions:
        if lang == 'de':
            ### both tasks together
            current_cond = [l for l in lines if l[header.index('condition')]==c]
            subjects = [int(l[header.index('subject')]) for l in current_cond]
            log_rts = [numpy.log10(float(l[header.index('log_rt')])) for l in current_cond]
            #log_rts = [float(l[header.index('rt')]) for l in current_cond]
            vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('task')])]
            test_vocab = test_vocab.union(set(vocab_w_ones))
            vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('stimulus')])]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            w_ones = [l[header.index('task')] for l in current_cond]
            w_twos = [l[header.index('stimulus')] for l in current_cond]
            sims[c]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
            for t in tasks:
                ### separate tasks
                current_cond = [l for l in lines if l[header.index('condition')]==c and l[header.index('task')]==t]
                subjects = [int(l[header.index('subject')]) for l in current_cond]
                log_rts = [numpy.log10(float(l[header.index('log_rt')])) for l in current_cond]
                exps = [l[header.index('expected_response')] for l in current_cond]
                #log_rts = [float(l[header.index('rt')]) for l in current_cond]
                vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('task')])]
                test_vocab = test_vocab.union(set(vocab_w_ones))
                vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('stimulus')])]
                test_vocab = test_vocab.union(set(vocab_w_twos))
                w_ones = [l[header.index('task')] for l in current_cond]
                w_twos = [l[header.index('stimulus')] for l in current_cond]
                sims['{}_{}'.format(t, c)]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
        else:
            full_sims[c] = list()
            for t in tasks:
                full_sims['{}_{}'.format(t, c)] = list()
    if lang == 'de':
        full_sims = reorganize_tms_sims(sims)
    return full_sims, test_vocab, prototypes

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
            #log_rts = [float(l[header.index('rt')]) for l in current_cond]
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

def read_fern(lang, trans_from_en):
    sims = {1 : dict(), 2 : dict()}
    test_vocab = set()
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
    if lang != 'en':
        print(lang)
        trans_vocab = set()
        for w in test_vocab:
            try:
                trs_w = trans_from_en[lang][w]
            except KeyError:
                print(w)
                continue
            trans_vocab = trans_vocab.union(trs_w)
        del test_vocab
        test_vocab = trans_vocab.copy()

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
            line = l.lower().strip().split(sep)
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
            line = l.lower().strip().split(sep)
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
                line = l.lower().strip().split()
                key = tuple(sorted([line[0], line[1]]))
                norm_key = set([m for k in key for w in k for m in transform_german_word(k)])
                test_vocab = test_vocab.union(norm_key)
                val = float(line[2].replace(',', '.'))
                sims[key] = val
    return sims, test_vocab

def read_mitchell(lang):
    assert lang == 'en'
    dimensions = list()
    with open(os.path.join('..', 'data', 'mitchell_dimensions.tsv')) as i:
        for l in i:
            line = l.strip().split()
            assert len(line) >= 2
            dimensions.append(line)
    assert len(dimensions) == 25

    return dimensions

languages = [
             'en',
             'de',
             'it',
             ]
senses = ['auditory', 'gustatory', 'haptic', 'olfactory', 'visual', 'hand_arm']   
print('loading original lancasted ratings...')
lancaster_ratings = {
                     'en' : read_lancaster_ratings(),
                     }
trans_from_en = dict()
### german translation
print('loading translations of ratings...')
for lang in tqdm(['de', 'it']):
    print(lang)
    lancaster_ratings[lang] = dict()
    trans_from_en[lang] = dict()
    with tqdm() as counter:
        with open(os.path.join('..', 'translations', 'lanc_fern_{}_to_en.tsv'.format(lang))) as i:
            for l in i:
                line = l.strip().split('\t')
                if lang == 'de':
                    for w in transform_german_word(line[0].lower()):
                        ### ratings
                        try:
                            lancaster_ratings['de'][w] = lancaster_ratings['en'][line[1]]
                        except KeyError:
                            pass
                        ### translations
                        try:
                            trans_from_en['de'][line[1]].add(w)
                        except KeyError:
                            trans_from_en['de'][line[1]] = {w}
                        counter.update(1)
                elif lang == 'it':
                    for w in [line[0].lower(), line[0].capitalize()]:
                        ### ratings
                        try:
                            lancaster_ratings['it'][w] = lancaster_ratings['en'][line[1]]
                        except KeyError:
                            pass
                        ### translations
                        try:
                            trans_from_en['it'][line[1]].add(w)
                        except KeyError:
                            trans_from_en['it'][line[1]] = {w}
                        counter.update(1)

rows = dict()
datasets = dict()
print('loading datasets...')
for lang in tqdm(languages):
    men, men_vocab = read_men(lang)
    simlex, simlex_vocab = read_simlex(lang)
    ws353, ws353_vocab = read_ws353(lang)
    fern_one, fern_two, fern_vocab = read_fern(lang, trans_from_en)
    germ_tms_ifg, germ_tms_ifg_vocab = read_german_ifg_tms(lang)
    de_tms_pipl, de_tms_pipl_vocab, prototypes = read_german_pipl_tms(lang)
    ita_tms_cereb, ita_tms_cereb_vocab = read_italian_cereb_tms(lang)
    basic_vocab = men_vocab.union(
                                  simlex_vocab,
                                  ws353_vocab,
                                  fern_vocab,
                                  germ_tms_ifg_vocab,
                                  de_tms_pipl_vocab,
                                  ita_tms_cereb_vocab,
                                  )
    rows[lang] = sorted(basic_vocab)
    datasets[lang] = dict()
    for dataset_name, dataset, proto in [
                    ### similarity/relatedness
                    ('simlex999-sim', simlex, {}),
                    ('ws353', ws353, {}),
                    ('men', men, {}), 
                    ### semantic network brain RSA
                    ('fern1', fern_one, {}),
                    ('fern2', fern_two, {}),
                    ## german TMS
                    ('de_sem-phon_tms_vertex', germ_tms_ifg['vertex'], {}),
                    ('de_sem-phon_tms_pIFG', germ_tms_ifg['pIFG'], {}),
                    ('de_sem-phon_tms_aIFG', germ_tms_ifg['aIFG'], {}),
                    #('de_sound-act_tms_all-pIPL', de_tms_pipl['pIPL'], prototypes),
                    #('de_sound-act_tms_all-sham', de_tms_pipl['sham'], prototypes),
                    #('de_sound-act_tms_soundtask-sham', de_tms_pipl['Geraeusch_sham'], prototypes),
                    #('de_sound-act_tms_actiontask-sham', de_tms_pipl['Handlung_sham'], prototypes),
                    #('de_sound-act_tms_soundtask-pIPL', de_tms_pipl['Geraeusch_pIPL'], prototypes),
                    #('de_sound-act_tms_actiontask-pIPL', de_tms_pipl['Handlung_pIPL'], prototypes),
                    ## italian TMS
                    ('it_distr-learn_tms_cereb', ita_tms_cereb['cedx'], {}),
                    ('it_distr-learn_tms_vertex', ita_tms_cereb['cz'], {}),
                    ]:
        datasets[lang][dataset_name] = (dataset, proto)

'''
results = dict()
results_file = 'cn_evaluation.tsv'
if os.path.exists(results_file):
    with open(results_file) as i:
        for l in i:
            line = l.strip().split('\t')
            assert len(line) > 3
            if line[0] not in results.keys():
                results[line[0]] = dict()
            if line[1] not in results[line[0]].keys():
                results[line[0]][line[1]] = dict()
            results[line[0]][line[1]][line[2]] = numpy.array(line[3:], dtype=numpy.float32)
'''

#fasttext_only = True
fasttext_only = False

models = dict()
vocabs = dict()
for lang in languages:
    #if lang not in results.keys():
    #    results[lang] = dict()
    models[lang] = dict()
    vocabs[lang] = dict()
    print('\n{}\n'.format(lang))
    for case in [
                 #'fasttext',
                 #'fasttext_aligned',
                 #'conceptnet',
                 ]:
        #if case in results[lang].keys():
        #    continue
        print('loading {}'.format(case))
        if case == 'fasttext':
            model = fasttext.load_model('../../../dataset/word_vectors/{}/cc.{}.300.bin'.format(lang, lang))
            vocab = model.words
        elif case == 'conceptnet':
            with open(os.path.join('..', '..', '..', 'dataset',
                                   'word_vectors', lang,
                                   'conceptnet_{}.pkl'.format(lang)
                                   ), 'rb') as i:
                model = pickle.load(i)
            vocab = model.keys()
        elif case == 'fasttext_aligned':
            with open(os.path.join('..', 'pickles', 
                                   'ft_{}_aligned.pkl'.format(lang)
                                   ), 'rb') as i:
                model = pickle.load(i)
            vocab = model.keys()
        model = {w : model[w] for w in vocab}
        vocab = [w for w in vocab]
        print('getting results for {}...'.format(case))
        #results = test_model(lang, case, model, vocab, results, datasets, trans_from_en)
        test_model(lang, case, model, vocab, datasets, trans_from_en)
    if fasttext_only:
        continue
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
        ### mitchell
        for row_mode in [
                         '', 
                         #'rowincol',
                         ]:
            if lang == 'en':
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
                test_model(lang, key, model, curr_vocab, datasets, trans_from_en)
                #results = test_model(lang, key, model, curr_vocab, results, datasets)
        ### lancaster
        filt_ratings = {w : freqs[w] for w in lancaster_ratings[lang].keys() if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0}
        sorted_ratings = [w[0] for w in sorted(filt_ratings.items(), key=lambda item: item[1], reverse=True)]
        filt_perc = {w : v['minkowski3'] for w, v in lancaster_ratings[lang].items() if w in vocab.keys() and w in freqs.keys() and vocab[w]!=0}
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
            for row_mode in [
                             '', 
                             'rowincol',
                             ]:
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
                    test_model(lang, key, model, curr_vocab, datasets, trans_from_en)
                    #results = test_model(lang, key, model, curr_vocab, results, datasets, trans_from_en)
        ### top-n frequencies
        filt_freqs = {w : f for w, f in freqs.items() if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0}
        sorted_freqs = [w[0] for w in sorted(filt_freqs.items(), key=lambda item: item[1], reverse=True)]
        for freq in tqdm([
                          100, 200, 500, 750,
                          1000, 
                          2500, 
                          5000, 7500,
                          10000, 12500, 15000, 17500,
                          20000, 25000
                          ]):
            for row_mode in [
                             '', 
                             'rowincol',
                             ]:
                for selection_mode in [
                                       'top', 
                                       'random',
                                       ]: 
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
                    test_model(lang, key, model, curr_vocab, datasets, trans_from_en)
