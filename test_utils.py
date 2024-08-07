import argparse
import fasttext
#import gensim
import numpy
import os
import pickle
import random
import scipy

from scipy import spatial

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
        '''
        if 'tms' in dataset_name:
            real.append(1 - v)
        elif 'decision' in dataset_name:
            real.append(1 - v)
        elif 'naming' in dataset_name:
            real.append(1 - v)
        else:
            real.append(v)
        '''
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
            current_pred = list()
            for w in w_ones:
                for w_two in w_twos:
                    ### always using dissimilarity
                    partial_pred = scipy.spatial.distance.cosine(model[w], model[w_two])
                    #partial_pred = 1 - scipy.stats.spearmanr(model[w], model[w_two]).statistic
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
        corr = list()
        missing_words = set()
        ### bootstrapping/iterations should be hard-coded now...
        for iter_dataset in tqdm(dataset):
            for s, s_data in iter_dataset.items():
                curr_corr, new_miss = compute_corr(iter_dataset, dataset_name, present_words, prototypes, trans_from_en)
                missing_words = missing_words.union(new_miss)
                if corr == None:
                    print('error with {}'.format([lang, case, dataset_name]))
                    continue
                corr.append(curr_corr)

        '''
        if 'tms' not in dataset_name:
            if 'behav' not in dataset_name and dataset_name[:2] not in ['1-', '2-']:
                corr, missing_words = compute_corr(dataset, dataset_name, present_words, prototypes, trans_from_en)
                if corr == None:
                    print('error with {}'.format([lang, case, dataset_name]))
                    continue
                corr = [corr]
            else:
                corr = list()
                all_dataset = [(k, v) for k, v in dataset.items()]
                for _ in tqdm(range(1000)):
                    ### bootstrap with subsampling
                    missing_words = set()
                    k = 100
                    chosen_items = random.sample(range(len(all_dataset)), k=k)
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
        '''
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
        os.makedirs(out_folder, exist_ok=True)
        out_f = os.path.join(out_folder, '{}.tsv'.format(dataset_name))
        write_res(out_f, lang, case, dataset_name, corr)
        print(out_f)
        #return results

def bootstrapper(full_data, residualize=False):
    ### bootstrapping with 1/4 of the original data
    proportion = 0.25
    ### labels
    labels = list(full_data.keys())
    ### subjects
    all_subjects = [list(v.keys()) for v in full_data.values()]
    # checking
    n_subjects = set([len(v) for v in all_subjects])
    assert len(n_subjects) == 1
    n_subjects = list(n_subjects[0])
    subjects = list(set([val for v in all_subjects for val in v]))
    assert len(subjects) == n_subjects
    n_iter_sub = int(n_subjects*proportion)
    ### here we create 1000
    boot_data = {l : list() for l in labels}
    for _ in range(1000):
        iter_subs = random.sample(subjects, k=n_iter_sub)
        iter_data_keys = {l : 
                               {s : random.sample(
                                                 full_data[l][s].keys(), 
                                                 k=int(len(full_data[l][s].keys())*proportion)
                                                 ) for s in iter_subs}
                                                 for l in labels}
        iter_data = {l : {s : {k : full_data[l][s][k] for k in iter_data_keys[l][s]} for s in iter_subs} for l in labels}
        ### residualization
        if residualization:
            struct_train_data = {l : {s : {k : full_data[l][s][k] for k in full_data[l][s].keys() if k not in iter_data_keys[l][s]} for s in iter_subs} for l in labels}
            flat_train_data = [(l, s, rt) for l, l_res in struct_train_data.items() for s, s_data in l_res.items() for k, rt in s_res.items()]
            flat_test_data = [(l, s, k, rt) for l, l_res in iter_data.items() for s, s_data in l_res.items() for k, rt in s_res.items()]
            model = sklearn.linear_model.LinearRegression()
            model.fit(
                      [[t[0],] for t in flat_train_data],
                      [[t[2]] for t in flat_train_data],
                      )
            preds = model.predict(
                                  [[t[0],] for t in flat_test_data]
                                  )
            residuals = [(real[0], real[1], real[2], real[3]-pred[0]) for real, pred in zip(flat_test_data, preds)]
            for l, s, k, r in residuals:
                iter_data[l][s][k] = r
        for l, l_data in iter_data.items():
            boot_data[l].append(l_data)
    return boot_data


def check_args(args):
    ### checking language if in first part of the name
    if '_' in args.dataset:
        assert args.dataset.split('_')[0] == args.language
    if args.dataset == 'men' and args.language != 'en':
        raise AssertionError

def load_dataset(args):
    if args.dataset == 'en_men':
        men, vocab = read_men()
    if '999' in args.dataset:
        simlex, vocab = read_simlex(lang)
    if '353' in args.dataset:
        ws353, vocab = read_ws353(lang)
    if 'fern' in args.dataset:
        if 'all' in args.dataset:
            fern, vocab = read_fern(lang, trans_from_en)
        elif 'categories' in args.dataset:
            fern_cats, vocab = read_fern_categories(lang, trans_from_en)
    if 'dirani' in args.dataset:
        dirani_n400_words, dirani_n400_pictures, vocab = read_dirani_n400(lang)
        if 'words' in args.dataset:
            data = {k : v for k, v in dirani_n400_words.items()}
        elif 'pictures' in args.dataset:
            data = {k : v for k, v in dirani_n400_pictures.items()}
    if 'abstract' in args.dataset:
        data, vocab = read_abstract_ipc()
    if 'de_behav' in args.dataset:
        data, vocab = read_german_behav()
    if 'it_behav' in args.dataset:
        data, vocab = read_italian_behav()
    if 'sem-phon' in args.dataset:
        data, vocab = read_de_sem_phon_tms()
    if 'sound-act' in args.dataset:
        data, vocab, prototypes = read_de_sound_act_tms()
    if 'distr-learn' in args.dataset:
        #related_ita_tms_cereb, unrelated_ita_tms_cereb, all_ita_tms_cereb, vocab = read_it_distr_learn_tms()
        data, vocab = read_it_distr_learn_tms()
    '''
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
                    #('it_distr-learn_related_tms_vertex', related_ita_tms_cereb['cz'], {}),
                    #('it_distr-learn_unrelated_tms_cereb', unrelated_ita_tms_cereb['cedx'], {}),
                    #('it_distr-learn_unrelated_tms_vertex', unrelated_ita_tms_cereb['cz'], {}),
                    #('it_distr-learn-bootstrap_all_tms_cereb', all_ita_tms_cereb['cedx'], {}),
                    #('it_distr-learn-bootstrap_all_tms_vertex', all_ita_tms_cereb['cz'], {}),
                    #('it_distr-learn-bootstrap_related_tms_cereb', related_ita_tms_cereb['cedx'], {}),
                    #('it_distr-learn-bootstrap_related_tms_vertex', related_ita_tms_cereb['cz'], {}),
                    #('it_distr-learn-bootstrap_unrelated_tms_cereb', unrelated_ita_tms_cereb['cedx'], {}),
                    #('it_distr-learn-bootstrap_unrelated_tms_vertex', unrelated_ita_tms_cereb['cz'], {}),
                    ]:
        datasets[lang][dataset_name] = (dataset, proto)
    ### fern categories
    #for dataset_name, dataset in fern_cats.items():
    #    datasets[lang][dataset_name] = (dataset, {})
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
                         #'matched_excl',
                         #'matched_excl-topten',
                         #'matched_excl-topfifty',
                         #'matched_non_excl',
                         #'matched_non_excl-topten',
                         #'matched_non_excl-topfifty',
                         #'opposite_excl',
                         #'opposite_excl-topten',
                         #'opposite_excl-topfifty',
                         #'opposite_non_excl',
                         #'opposite_non_excl-topten',
                         #'opposite_non_excl-topfifty',
                         ]:
                curr_dataset_name = '{}#{}'.format(dataset_name, poss)
                datasets[lang][curr_dataset_name] = (dataset, proto)
    '''

    return vocab, datasets

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
                                'simlex999-sim',
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
                                'de_sem-phon-bootstrap',
                                'de_sound-act',
                                'de_sound-act-bootstrap',
                                'it_distr-learn',
                                'it_distr-learn-bootstrap',
                                ],
                        required=True,
                        )
    #senses = ['auditory', 'gustatory', 'haptic', 'olfactory', 'visual', 'hand_arm']   
    args = parser.parse_args()
    check_args(args)

    return args
