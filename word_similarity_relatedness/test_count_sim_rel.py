import argparse
import fasttext
#import gensim
import numpy
import os
import pickle
import random
import scipy
import sys
import time

#from gensim import downloader
from scipy import spatial, stats
from tqdm import tqdm

sys.path.append('..')

from utf_utils import transform_german_word
from utils import build_ppmi_vecs, read_lancaster_ratings

def check_dataset_words(dataset_name, dataset, present_words, trans_from_en, prototyping=False, printing=False):
    #print('checking if words appear in the dictionary...')
    missing_words = set()
    test_sims = list()
    if prototyping:
        dataset = [((w, w), 0)  for w in dataset] 
    #if not prototyping and printing==False:
    #    #with tqdm() as counter:
    #    counter = tqdm()
    if type(dataset) != list:
        dataset = [(k, v) for k, v in dataset.items()]
    for ws, val in dataset:
        marker = True
        w_ones = list()
        w_twos = list()
        if 'fern' in dataset_name and lang in ['de', 'it']:
            ### word one
            try:
                candidates = trans_from_en[lang][ws[0]]
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
                candidates = trans_from_en[lang][ws[1]]
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
        #if not prototyping and printing==False:
        #    counter.update(1)
        if marker:
            #test_sims[ws] = val
            test_sims.append((w_ones, w_twos, val))
    #if not prototyping:
    #    del counter
    #if len(missing_words) > 0:
    #    print('missing words: {}'.format(missing_words))
    return test_sims, missing_words

def compute_corr(dataset, dataset_name, present_words, prototypes, trans_from_en, printing=0):
    test_sims, missing_words = check_dataset_words(dataset_name, dataset, present_words, trans_from_en,)
    if len(prototypes.keys()) > 0:
        proto_vecs = prototypes
    if len(test_sims) == 0:
        #continue
        corr = None
        return corr
    real = list()
    pred = list()
    #for k, v in test_sims.items():
    for w_ones, w_twos, v in test_sims:
        ###CHECK
        #if 'tms' or 'behav' in dataset_name:
        #    ### for reaction times, it's the opposite
        #    real.append(-v)
        #real.append(v)
        if 'tms' in dataset_name:
            real.append(1 - v)
        elif 'decision' in dataset_name:
            real.append(1 - v)
        elif 'naming' in dataset_name:
            real.append(1 - v)
        else:
            real.append(v)
        ### all possible transformations...
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
                         'opposite_non_excl',
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
    return corr, missing_words

def write_res(out_f, lang, case, dataset_name, corr, sleep=False):
    with open(out_f, 'w') as o:
        o.write('{}\t{}\t{}\t'.format(lang, case, dataset_name))
        for c in corr:
            o.write('{}\t'.format(c))
        #o.write('\n')
        if sleep == True:
            time.sleep(10)

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
        ### for fasttext we only use uppercase!
        if w[0].isupper() == False and model=='fasttext':
            continue
        try:
            vocab.index(w)
        except ValueError:
            continue
        present_words.append(w)
    for dataset_name, dataset_proto in datasets[lang].items():
        dataset = dataset_proto[0]
        prototypes = dataset_proto[1]
        if len(prototypes.keys())>0:
            proto_sims = dict()
            for k, v in prototypes.items():
                ws, _ = check_dataset_words(dataset_name, v, present_words, trans_from_en, prototyping=True)
                proto_sims[k] = set([w for _ in ws for w in _[0]])
            del prototypes
            prototypes = dict()
            for k, v in proto_sims.items():
                print(k)
                print(v)
                assert len(v) > 0
                current_vecs = list()
                for w in v:
                    current_vecs.append(model[w])
                current_vec = numpy.average(current_vecs, axis=0)
                prototypes[k] = current_vec
        if len(dataset) == 0:
            continue
        #if case not in results[lang].keys():
        #    results[lang][case] = dict()
        if 'tms' not in dataset_name:
            if 'behav' not in dataset_name:
                corr, missing_words = compute_corr(dataset, dataset_name, present_words, prototypes, trans_from_en)
                if corr == None:
                    print('error with {}'.format([lang, case, dataset_name]))
                    continue
                corr = [corr]
            else:
                '''
                if 'bootstrap' not in dataset_name:
                    #all_dataset = [(k, v) for k, v in dataset.items()]
                    corr, missing_words = compute_corr(dataset, dataset_name, present_words, prototypes, trans_from_en)
                    if corr == None:
                        print('error with {}'.format([lang, case, dataset_name]))
                        continue
                    corr = [corr]
                else:
                '''
                corr = list()
                all_dataset = [(k, v) for k, v in dataset.items()]
                for _ in tqdm(range(1000)):
                    ### bootstrap with subsampling
                    missing_words = set()
                    chosen_items = random.sample(range(len(all_dataset)), k=100)
                    curr_data = [all_dataset[i] for i in chosen_items]
                    curr_corr, new_miss = compute_corr(curr_data, dataset_name, present_words, prototypes, trans_from_en,)
                    missing_words = missing_words.union(new_miss)
                    if curr_corr == None:
                        print('error with {} - subject {}'.format([lang, case, dataset_name, _]))
                    else:
                        corr.append(curr_corr)
                    #printing += 1
        else:
            corr = list()
            printing = 0
            if 'bootstrap' not in dataset_name:
                for s, s_data in dataset.items():
                    missing_words = set()
                    curr_corr, new_miss = compute_corr(s_data, dataset_name, present_words, prototypes, trans_from_en, printing=printing)
                    missing_words = missing_words.union(new_miss)
                    if curr_corr == None:
                        print('error with {} - subject {}'.format([lang, case, dataset_name, s]))
                    else:
                        corr.append(curr_corr)
                    printing += 1
            ### bootstrapping 1000 data splits
            else:
                for _ in tqdm(range(1000)):
                    ### hierarchical bootstrap with subsampling
                    chosen_subs = random.sample(list(dataset.keys()), k=int(len(dataset.keys())/4))
                    #all_data = [(k, v) for sub in chosen_subs for k, v in dataset[sub].items()]
                    #curr_data = random.choices(all_data, k=int(len(all_data)/10))
                    missing_words = set()
                    curr_data = list()
                    for sub in chosen_subs:
                        possibilities = [(k, v) for k, v in dataset[sub].items()]
                        chosen_items = random.sample(range(len(possibilities)), k=len(possibilities))
                        curr_data.extend([possibilities[i] for i in chosen_items])
                    curr_corr, new_miss = compute_corr(curr_data, dataset_name, present_words, prototypes, trans_from_en, printing=printing)
                    missing_words = missing_words.union(new_miss)
                    if curr_corr == None:
                        print('error with {} - subject {}'.format([lang, case, dataset_name, _]))
                    else:
                        corr.append(curr_corr)
                    printing += 1
            #corr = numpy.nanmean(corrs)
        print('\n')
        print('{} model'.format(case))
        print('correlation with {} dataset:'.format(dataset_name))
        print(numpy.nanmean(corr))
        if len(missing_words) > 0:
            print('missing words: {}'.format(missing_words))
        corpus_fold = case.split('_')[1] if 'ppmi' in case else case
        details = '_'.join(case.split('_')[2:]) if 'ppmi' in case else case
        out_folder = os.path.join(
                                  'sim-rel_results',
                                  lang, 
                                  corpus_fold, 
                                  details,
                                  )
        try:
            os.makedirs(out_folder, exist_ok=True)
        except PermissionError:
            if os.path.exists(out_folder):
                pass
            else:
                time.sleep(300)
                os.makedirs(out_folder, exist_ok=True)
        out_f = os.path.join(out_folder, '{}.tsv'.format(dataset_name))
        try:
            write_res(out_f, lang, case, dataset_name, corr)
        except PermissionError:
            time.sleep(300)
            write_res(out_f, lang, case, dataset_name, corr, sleep=True)
        print(out_f)
        #return results

def read_italian_cereb_tms(lang):
    lines = list()
    with open(os.path.join('..', 'tms', 'data', 'italian_tms_cereb.tsv')) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = [w for w in line]
                continue
            lines.append(line)
    conds = set([l[header.index('condition')] for l in lines])
    all_sims = dict()
    all_full_sims = dict()
    related_sims = dict()
    related_full_sims = dict()
    unrelated_sims = dict()
    unrelated_full_sims = dict()
    test_vocab = set()
    for name in conds:
        if lang == 'it':
            for m_i, marker in enumerate(['1', '0', 'all']):
                if m_i < 2:
                    current_cond = [l for l in lines if l[header.index('condition')]==name and l[header.index('Meaningful')]==marker]
                else:
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
                if m_i == 0:
                    related_sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
                elif m_i == 1:
                    unrelated_sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
                elif m_i == 2:
                    all_sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
        else:
            related_full_sims[name] = list()
            unrelated_full_sims[name] = list()
            all_full_sims[name] = list()
    if lang == 'it':
        related_full_sims = reorganize_tms_sims(related_sims)
        unrelated_full_sims = reorganize_tms_sims(unrelated_sims)
        all_full_sims = reorganize_tms_sims(all_sims)
    return related_full_sims, unrelated_full_sims, all_full_sims, test_vocab

def read_german_pipl_tms(lang):
    ### first reading ratings
    ratings = dict()
    with open(os.path.join('..', 'tms', 'data', 'phil_annotated_ratings_v9.tsv')) as i:
        for l_i, l in enumerate(i):
            line = l.replace(',', '.').strip().split('\t')
            if l_i == 0:
                header = [w for w in line]
                continue
            ratings[line[0]] = dict()
            sound = line[header.index('Geraeusch')]
            try:
                sound = float(sound)
                ratings[line[0]]['sound'] = sound
            except ValueError:
                pass
            action = line[header.index('Handlung')]
            try:
                action = float(action)
                ratings[line[0]]['action'] = action
            except ValueError:
                pass
            if len(ratings[line[0]].keys()) == 0:
                print(line[0])
                del ratings[line[0]]
    ### topten
    sims = dict()
    test_vocab = set()
    lines = list()
    ### possibilities in task-modelling:
    # centroid overall (all)
    # both positive (both_pos)
    # both negative (both_neg)
    # matched exclusive (action_pos_sound_neg, sound_pos_action_neg)
    # matched non-exclusive (action_pos, sound_pos)
    prototypes = {
                  'action_pos' : list(), 
                  'sound_pos' : list(),
                  'action_neg' : list(),
                  'sound_neg' : list(), 
                  'all' : list(),
                  'both_pos' : list(),
                  'both_neg' : list(),
                  'action_pos_sound_neg' : list(), 
                  'sound_pos_action_neg' : list(),
                  }
    with open(os.path.join('..', 'tms', 'data', 'de_tms_pipl.tsv')) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = [w for w in line]
                continue
            missing = list()
            w = line[header.index('stimulus')]
            if line[header.index('sound_word')] != 'NA':
                if w not in ratings.keys():
                    missing.append(w)
            assert len(missing) == 0
            if line[header.index('action_word')] == '1' and line[header.index('sound_word')] == '-1':
                prototypes['action_pos_sound_neg'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '-1' and line[header.index('sound_word')] == '1':
                prototypes['sound_pos_action_neg'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '1' and line[header.index('sound_word')] == '1':
                prototypes['both_pos'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '-1' and line[header.index('sound_word')] == '-1':
                prototypes['both_neg'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '1':
                prototypes['action_pos'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '-1':
                prototypes['action_neg'].append(line[header.index('stimulus')])
            if line[header.index('sound_word')] == '1':
                prototypes['sound_pos'].append(line[header.index('stimulus')])
            if line[header.index('sound_word')] == '-1':
                prototypes['sound_neg'].append(line[header.index('stimulus')])
            #if line[header.index('action_word')] == '-1' and line[header.index('sound_word')] == '1':
            #    prototypes['action'].append(line[header.index('stimulus')])
            #if line[header.index('sound_word')] == '-1' and line[header.index('action_word')] == '1':
            #    prototypes['sound'].append(line[header.index('stimulus')])
            if 'lexical_decision' in line:
                continue
            prototypes['all'].append(line[header.index('stimulus')])
            lines.append(line)
    prototypes = {k : set(v) for k, v in prototypes.items()}
    top_tenned = [
          'action_pos', 
          'sound_pos',
          'both_pos',
          'action_pos_sound_neg', 
          'sound_pos_action_neg',
          ]
    for tenned in top_tenned:
        ### top ten percent
        for percent, mult in [('ten', 0.1), ('fifty', 0.5)]:
            ten_percent = int(len(prototypes[tenned])*mult)
            if 'sound_pos' in tenned:
                top = [s[0] for s in sorted([(k, ratings[k]['sound']) for k in prototypes[tenned]], key=lambda item : item[1], reverse=True)][:ten_percent]
            if 'action_pos' in tenned:
                top = [s[0] for s in sorted([(k, ratings[k]['action']) for k in prototypes[tenned]], key=lambda item : item[1], reverse=True)][:ten_percent]
            else:
                top_s = [s[0] for s in sorted([(k, ratings[k]['sound']) for k in prototypes[tenned]], key=lambda item : item[1], reverse=True)]
                top_a = [s[0] for s in sorted([(k, ratings[k]['action']) for k in prototypes[tenned]], key=lambda item : item[1], reverse=True)]
                top = [s[0] for s in sorted([(k, top_a.index(k)+top_s.index(k)) for k in prototypes[tenned]], key=lambda item : item[1])][:ten_percent]
            print(top)
            prototypes['{}-top{}'.format(tenned, percent)] = top

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

def read_italian_behav(lang):
    ### lexical decition times
    sims = {'word-naming' : dict()}
    test_vocab = set()
    if lang == 'it':
        for case in sims.keys(): 
            sims[case] = dict()
            measures = dict()
            with open(os.path.join('..', 'behavioural', 'data', 'varlex_{}_it.tsv'.format(case))) as i:
                for l_i, l in enumerate(i):
                    line = l.replace(',', '.').strip().split('\t')
                    if l_i < 2:
                        continue
                    word = line[0].lower()
                    test_vocab = test_vocab.union(set([word, word.capitalize()]))
                    word_rt = float(line[-1])
                    measures[word] = word_rt
            for k_one_i, k_one in enumerate(sorted(measures.keys())):
                for k_two_i, k_two in enumerate(sorted(measures.keys())):
                    if k_two_i <= k_one_i:
                        continue
                    key = tuple(sorted([k_one, k_two]))
                    sims[case][key] = abs(measures[k_one]-measures[k_two])
    return sims, test_vocab

def read_german_behav(lang):
    ### lexical decition times
    sims = {'word-naming' : dict(), 'lexical-decision' : dict()}
    test_vocab = set()
    if lang == 'de':
        for case in sims.keys(): 
            measures = dict()
            with open(os.path.join('..', 'behavioural', 'data', 'devel_{}_de.tsv'.format(case))) as i:
                for l_i, l in enumerate(i):
                    line = l.replace(',', '.').strip().split('\t')
                    if l_i == 0:
                        header = [w for w in line]
                        marker = 'rt' if 'lex' in case else 'on'
                        ya = header.index('{}.ya.m'.format(marker))
                        oa = header.index('{}.oa.m'.format(marker))
                        continue
                    if len(line) != len(header):
                        print(line)
                        continue
                    word = line[0].lower()
                    versions = transform_german_word(word)
                    test_vocab = test_vocab.union(versions)
                    word_rt = float(float(line[ya])+float(line[oa]))/2
                    measures[word] = word_rt
            for k_one_i, k_one in enumerate(sorted(measures.keys())):
                for k_two_i, k_two in enumerate(sorted(measures.keys())):
                    if k_two_i <= k_one_i:
                        continue
                    key = tuple(sorted([k_one, k_two]))
                    sims[case][key] = abs(measures[k_one]-measures[k_two])
    return sims, test_vocab

def read_german_ifg_tms(lang):
    sims = dict()
    test_vocab = set()
    lines = list()
    with open(os.path.join('..', 'tms', 'data', 'original', 'de_tms_sem-phon_ifg.tsv')) as i:
        for l_i, l in enumerate(i):
            line = l.replace(',', '.').strip().split('\t')
            if 'NA' in line:
                print(line)
                continue
            if l_i == 0:
                header = [w for w in line]
                continue
            #assert len(line)==len(header)
            if '' in line:
                continue
            if len(line) < len(header)-1:
                print(line)
                continue
            lines.append(line)
    ###
    conditions = set([l[header.index('stim')] for l in lines])
    tasks = set([l[header.index('task')] for l in lines])
    full_sims = dict()
    #for c, name in conditions.items():
    for t in tasks:
        for c in conditions:
            name = '{}-{}'.format(c, t)
            #print(name)
            if lang == 'de':
                ###One participant was replaced due to an overall mean error rate of 41.8% - sub 3
                #current_cond = [l for l in lines if l[header.index('stim')] in c and int(l[header.index('subj')])!=3]
                current_cond = [l for l in lines if l[header.index('stim')] in name and l[header.index('task')] in name and int(l[header.index('subj')])!=3]
                        #and l[header.index('utterance')]!='NA']
                tasks = [l[header.index('task')] for l in current_cond]
                assert len(set(tasks)) == 1
                subjects = [int(l[header.index('subj')]) for l in current_cond]
                assert len(set(subjects)) == 24
                #print(subjects)
                rts = [float(l[header.index('RT')]) for l in current_cond]
                log_rts = [numpy.log10(float(l[header.index('RT')])) for l in current_cond]
                vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('utterance')])]
                test_vocab = test_vocab.union(set(vocab_w_ones))
                vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('item')].split('.')[0])]
                test_vocab = test_vocab.union(set(vocab_w_twos))
                w_ones = [l[header.index('utterance')] for l in current_cond]
                w_twos = [l[header.index('item')].split('.')[0] for l in current_cond]
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

def read_dirani_n400(lang):
    sims = {'words' : dict(), 'pictures' : dict()}
    if lang != 'en':
        trans_path = os.path.join(
                                 '..', 
                                 'meg-dirani',
                                 'meg-dirani-n400_stimuli_translations_it_de.tsv'
                                 )
        trans = dict()
        with open(trans_path) as i:
            for l_i, l in enumerate(i):
                line = l.strip().split('\t')
                if l_i == 0:
                    rel_trans = line.index('word_{}'.format(lang))
                    continue
                trans[line[0].strip()] = line[rel_trans].strip()

    test_vocab = set()
    for dataset in sims.keys():
        sims[dataset] = dict()
        file_path = os.path.join(
                                 '..', 
                                 'meg-dirani',
                                 'data', 
                                 'reorganized_dataset',
                                 dataset,
                                 'tsv',
                                 '300-400ms_similarities.tsv')
        with open(file_path) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                w_one = line[0]
                w_two = line[1]
                ### translating if needed
                if lang != 'en':
                    w_one = trans[w_one]
                    w_two = trans[w_two]
                ### german will be transformed
                if lang != 'de':
                    test_vocab = test_vocab.union(set([w_one, w_two]))
                else:
                    test_vocab = test_vocab.union(transform_german_word(w_one))
                    test_vocab = test_vocab.union(transform_german_word(w_two))
                sims[dataset][(w_one, w_two)] = numpy.average(numpy.array(line[2:], dtype=numpy.float32))

    return sims['words'], sims['pictures'], test_vocab

def read_abstract_ipc(lang):
    sims = dict()
    test_vocab = set()
    if lang == 'de':
        file_path = os.path.join(
                                 '..', 
                                 'fmri',
                                 'data', 
                                 'de_abstract-ipc_data.tsv')
        with open(file_path) as i:
            for l_i, l in enumerate(i):
                line = l.strip().split('\t')
                w_one = line[0]
                w_two = line[1]
                test_vocab = test_vocab.union(transform_german_word(w_one))
                test_vocab = test_vocab.union(transform_german_word(w_two))
                assert len(line[2:]) == 19
                sims[(w_one, w_two)] = 1-numpy.average(numpy.array(line[2:], dtype=numpy.float32))

    return sims, test_vocab

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
             #'en',
             #'de',
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
                            print(w)
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
    dirani_n400_words, dirani_n400_pictures, dirani_vocab = read_dirani_n400(lang)
    de_abstract_ipc, de_abstract_ipc_vocab = read_abstract_ipc(lang)
    de_behav, de_behav_vocab = read_german_behav(lang)
    it_behav, it_behav_vocab = read_italian_behav(lang)
    germ_tms_ifg, germ_tms_ifg_vocab = read_german_ifg_tms(lang)
    de_tms_pipl, de_tms_pipl_vocab, prototypes = read_german_pipl_tms(lang)
    related_ita_tms_cereb, unrelated_ita_tms_cereb, all_ita_tms_cereb, ita_tms_cereb_vocab = read_italian_cereb_tms(lang)
    basic_vocab = men_vocab.union(
                                  simlex_vocab,
                                  ws353_vocab,
                                  fern_vocab,
                                  dirani_vocab,
                                  de_abstract_ipc_vocab,
                                  germ_tms_ifg_vocab,
                                  de_tms_pipl_vocab,
                                  de_behav_vocab,
                                  it_behav_vocab,
                                  ita_tms_cereb_vocab,
                                  )
    rows[lang] = sorted(basic_vocab)
    datasets[lang] = dict()
    for dataset_name, dataset, proto in [
                    ### similarity/relatedness
                    #('simlex999-sim', simlex, {}),
                    #('ws353', ws353, {}),
                    #('men', men, {}), 
                    ### semantic network brain RSA
                    #('fern1', fern_one, {}),
                    #('fern2', fern_two, {}),
                    ### EEG semantics RSA
                    #('dirani-n400-words', dirani_n400_words, {}),
                    #('dirani-n400-pictures', dirani_n400_pictures, {}),
                    ### german naming times
                    #('de_behav-word-naming', de_behav['word-naming'], {}),
                    #('de_behav-lexical-decision', de_behav['lexical-decision'], {}),
                    ### german abstract words
                    #('de_abstract-ipc', de_abstract_ipc, {}),
                    ## german TMS
                    #('de_sem-phon_tms_vertex', germ_tms_ifg['vertex-sem'], {}),
                    #('de_sem-phon_tms_pIFG', germ_tms_ifg['pIFG-sem'], {}),
                    #('de_sem-phon_tms_aIFG', germ_tms_ifg['aIFG-sem'], {}),
                    #('de_sem-phon-bootstrap_tms_vertex', germ_tms_ifg['vertex-sem'], {}),
                    #('de_sem-phon-bootstrap_tms_pIFG', germ_tms_ifg['pIFG-sem'], {}),
                    #('de_sem-phon-bootstrap_tms_aIFG', germ_tms_ifg['aIFG-sem'], {}),
                    ### italian naming times
                    #('it_behav-word-naming', it_behav['word-naming'], {}),
                    ## italian TMS
                    #('it_distr-learn_all_tms_cereb', all_ita_tms_cereb['cedx'], {}),
                    #('it_distr-learn_all_tms_vertex', all_ita_tms_cereb['cz'], {}),
                    #('it_distr-learn_related_tms_cereb', related_ita_tms_cereb['cedx'], {}),
                    #('it_distr-learn_related_tms_vertex', rewac_lancaster_freq_hi-perceptual__10000.0'lated_ita_tms_cereb['cz'], {}),
                    #('it_distr-learn_unrelated_tms_cereb', unrelated_ita_tms_cereb['cedx'], {}),
                    #('it_distr-learn_unrelated_tms_vertex', unrelated_ita_tms_cereb['cz'], {}),
                    #('it_distr-learn-bootstrap_all_tms_cereb', all_ita_tms_cereb['cedx'], {}),
                    ('it_distr-learn-bootstrap_all_tms_vertex', all_ita_tms_cereb['cz'], {}),
                    #('it_distr-learn-bootstrap_related_tms_cereb', related_ita_tms_cereb['cedx'], {}),
                    ('it_distr-learn-bootstrap_related_tms_vertex', related_ita_tms_cereb['cz'], {}),
                    #('it_distr-learn-bootstrap_unrelated_tms_cereb', unrelated_ita_tms_cereb['cedx'], {}),
                    ('it_distr-learn-bootstrap_unrelated_tms_vertex', unrelated_ita_tms_cereb['cz'], {}),
                    ]:
        datasets[lang][dataset_name] = (dataset, proto)
    for dataset_name, dataset, proto in [
            #('de_sound-act-bootstrap_tms_all-pIPL', de_tms_pipl['pIPL'], prototypes),
            #('de_sound-act-bootstrap_tms_all-sham', de_tms_pipl['sham'], prototypes),
            #('de_sound-act_tms_all-pIPL', de_tms_pipl['pIPL'], prototypes),
            #('de_sound-act_tms_all-sham', de_tms_pipl['sham'], prototypes),
            #('de_sound-act-bootstrap_tms_soundtask-sham', de_tms_pipl['Geraeusch_sham'], prototypes),
            #('de_sound-act-bootstrap_tms_actiontask-sham', de_tms_pipl['Handlung_sham'], prototypes),
            #('de_sound-act_tms_soundtask-sham', de_tms_pipl['Geraeusch_sham'], prototypes),
            #('de_sound-act_tms_actiontask-sham', de_tms_pipl['Handlung_sham'], prototypes),
            #('de_sound-act-bootstrap_tms_soundtask-pIPL', de_tms_pipl['Geraeusch_pIPL'], prototypes),
            #('de_sound-act-bootstrap_tms_actiontask-pIPL', de_tms_pipl['Handlung_pIPL'], prototypes),
            #('de_sound-act_tms_soundtask-pIPL', de_tms_pipl['Geraeusch_pIPL'], prototypes),
            #('de_sound-act_tms_actiontask-pIPL', de_tms_pipl['Handlung_pIPL'], prototypes),
            ]:
            ### possibilities in task-modelling:
            # centroid overall (all)
            # both positive (both_pos)
            # both negative (both_neg)wac_lancaster_freq_hi-perceptual__10000.0'
            # matched exclusive (action_pos_sound_neg, sound_pos_action_neg)
            # matched non-exclusive (action_pos, sound_pos)
            for poss in [
                         #'all', 
                         #'both_pos',
                         #'both_pos-topten',
                         #'both_pos-topfifty',
                         #'both_neg',
                         'matched_excl',
                         #'matched_excl-topten',
                         #'matched_excl-topfifty',
                         #'matched_non_excl',
                         #'matched_non_excl-topten',
                         #'matched_non_excl-topfifty',
                         #'opposite_excl',
                         #'opposite_non_excl',
                         ]:
                curr_dataset_name = '{}#{}'.format(dataset_name, poss)
                datasets[lang][curr_dataset_name] = (dataset, proto)

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
                 'fasttext_aligned',
                 'conceptnet',
                 ]:
        #if case in results[lang].keys():
        #    continue
        print('loading {}'.format(case))
        base_folder = os.path.join(
                                    '/',
                                    'data',
                                    'u_bruera_software',
                                    #'tu_bruera',
                                    'word_vectors', 
                                    lang, 
                                    )
        if case == 'fasttext':
            model = fasttext.load_model(
                                        os.path.join(
                                            base_folder,
                                            'cc.{}.300.bin'.format(lang)
                                            )
                                        )
            vocab = model.words
        elif case == 'conceptnet':
            with open(
                    os.path.join(
                        base_folder,
                       'conceptnet_{}.pkl'.format(lang)
                       ), 'rb') as i:
                model = pickle.load(i)
            vocab = model.keys()
        elif case == 'fasttext_aligned':
            with open(
                      os.path.join(
                                base_folder,
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
            if corpus == 'bnc':
                min_count = 10
            elif corpus == 'cc100':
                min_count = 500
            else:
                min_count = 100
        else:
            if corpus == 'cc100':
                min_count = 100
            else:
                min_count = 10
        with open(os.path.join(
                                '/',
                                'data',
                                #'u_bruera_software',
                                'tu_bruera',
                                'counts',
                               lang, corpus, 
                               '{}_{}_uncased_vocab_min_{}.pkl'.format(lang, corpus, min_count),
                               ), 'rb') as i:
            vocab = pickle.load(i)
        with open(os.path.join(
                                '/',
                                'data',
                                #'u_bruera_software',
                                'tu_bruera',
                                'counts',
                               lang, corpus, 
                               '{}_{}_uncased_word_freqs.pkl'.format(lang, corpus),
                               ), 'rb') as i:
            freqs = pickle.load(i)
        print('total size of the corpus: {:,} tokens'.format(sum(freqs.values())))
        with open(os.path.join(
                                '/',
                                'data',
                                #'u_bruera_software',
                                'tu_bruera',
                                'counts',
                               lang, corpus, 
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
                          2500, 
                          5000, 
                          7500,
                          10000, 12500, 15000, 17500,
                          20000, 25000
                          ]):
            for row_mode in [
                             '', 
                          #   'rowincol',
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
                        idxs = random.sample(range(len(sorted_ratings)), k=min(freq, len(sorted_ratings)))
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
                             #'rowincol',
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
                        idxs = random.sample(range(len(sorted_freqs)), k=min(freq, len(sorted_freqs)))
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
