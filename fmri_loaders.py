import numpy
import os
import re

from utf_utils import transform_basic_word, transform_german_word

def read_fern_areas(args, trans_from_en):

    required_dataset = int(re.sub('\D', '', args.dataset))
    #dis_sims = {1 : {'all' : dict()}, 2 : dict()}
    #dis_sims = {'{}_fern{}_{}#{}'.format(args.lang, required_dataset, args.stat_approach, area) : {k : v for k, v in dis_sims[required_dataset].items()} for area in [
    #            'L_IFG', 
    #            'L_PMTG', 
    #            'L_inferiorparietal'
    #            ]
    #            }
    dis_sims = dict()
    test_vocab = set()
    areas = [
            'L_IFG', 
            'L_PMTG', 
            'L_inferiorparietal'
            ]
    #for dataset in dis_sims.keys():
    #    #dis_sims[dataset] = {'all' : dict()}
    for area in areas:
        #area = dataset.split('#')[-1]
        key = '{}_fern{}_{}#{}'.format(args.lang, required_dataset, args.stat_approach, area)
        if required_dataset == 1:
            dis_sims[key] = {'all' : dict()}
        else:
            dis_sims[key] = dict()
        file_path = os.path.join(
                                 'data', 
                                 'fmri', 
                                 'fernandino', 
                                 'fern{}_{}_similarities.tsv'.format(required_dataset, area)
                                 )
        #print(file_path)
        with open(file_path) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                test_vocab = test_vocab.union(set(
                    transform_basic_word(line[0])+transform_basic_word(line[1])
                    ))
                if required_dataset == 1:
                    sim = numpy.average(numpy.array(line[2:], dtype=numpy.float32))
                    ### we want dissimilarity
                    dis_sims[key]['all'][(line[0], line[1])] = 1 - sim
                else:
                    for sub, sub_val in enumerate(line[2:]):
                        if sub not in dis_sims[key].keys():
                            dis_sims[key][sub] = dict()  
                        dis_sims[key][sub][(line[0], line[1])] = 1 - float(sub_val)
    if args.lang != 'en':
        print(args.lang)
        trans_vocab = set()
        for w in test_vocab:
            try:
                trs_w = trans_from_en[args.lang][w]
            except KeyError:
                print(w)
                continue
            trans_vocab = trans_vocab.union(trs_w)
        del test_vocab
        test_vocab = trans_vocab.copy()

    return dis_sims, test_vocab

def read_fern(args, trans_from_en):

    required_dataset = int(re.sub('\D', '', args.dataset))
    dis_sims = {1 : {'all' : dict()}, 2 : dict()}
    dis_sims = {'{}_fern{}_{}#all'.format(args.lang, args.stat_approach, required_dataset) : dis_sims[required_dataset]}
    test_vocab = set()
    for dataset in dis_sims.keys():
        #dis_sims[dataset] = {'all' : dict()}
        file_path = os.path.join(
                                 'data', 
                                 'fmri', 
                                 'fernandino', 
                                 'fern{}_semantic_network.tsv'.format(required_dataset)
                                 )
        with open(file_path) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                test_vocab = test_vocab.union(set(
                    transform_basic_word(line[0])+transform_basic_word(line[1])
                    ))
                if required_dataset == 1:
                    sim = numpy.average(numpy.array(line[2:], dtype=numpy.float32))
                    ### we want dissimilarity
                    dis_sims[dataset]['all'][(line[0], line[1])] = 1 - sim
                else:
                    for sub, sub_val in enumerate(line[2:]):
                        if sub not in dis_sims[dataset].keys():
                            dis_sims[dataset][sub] = dict()  
                        dis_sims[dataset][sub][(line[0], line[1])] = 1 - float(sub_val)
    if args.lang != 'en':
        print(args.lang)
        trans_vocab = set()
        for w in test_vocab:
            try:
                trs_w = trans_from_en[args.lang][w]
            except KeyError:
                print(w)
                continue
            trans_vocab = trans_vocab.union(trs_w)
        del test_vocab
        test_vocab = trans_vocab.copy()

    return dis_sims, test_vocab

def read_fern_categories(args, trans_from_en):
    required_dataset = int(re.sub('\D', '', args.dataset))
    mapper = dict()
    with open(os.path.join(
                           'data', 
                           'fmri', 
                           'fernandino', 
                           'fernandino_brain_concepts_categories.tsv')
                           ) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split('\t')
            first_cat = 'concrete'
            if 'abstract' in line[1]:
                first_cat = 'abstract'
            if 'mental' in line[1]:
                first_cat = 'abstract'
            if 'human_group' in line[1]:
                first_cat = 'abstract'
            if 'social_event' in line[1]:
                first_cat = 'abstract'
            mapper[line[0]] = (first_cat, line[1])
    assert len(mapper.keys()) == 524
            
    dis_sims = dict()
    test_vocab = set()
    for dataset in [required_dataset]:
        file_path = os.path.join(
                                 'data', 
                                 'fmri', 
                                 'fernandino', 
                                 'fern{}_semantic_network.tsv'.format(required_dataset)
                                 )
        with open(file_path) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                if line[0] not in mapper.keys():
                    print(line[0])
                    continue
                if line[1] not in mapper.keys():
                    print(line[1])
                    continue
                test_vocab = test_vocab.union(set(
                    transform_basic_word(line[0])+transform_basic_word(line[1])
                    ))
                ### abstract/concrete: idx 0
                ### finer grained categories: idx 1
                for idx in [
                            0, 
                            1,
                            ]:
                    cat = set([mapper[line[0]][idx], mapper[line[1]][idx]])
                    if len(cat) == 1:
                        marker = '{}_fern{}_{}#{}'.format(args.lang, required_dataset, args.stat_approach, list(cat)[0])
                        if marker not in dis_sims.keys():
                            dis_sims[marker] = {'all' : dict()}
                        ### we want dissimilarity
                        if required_dataset == 1:
                            sim = numpy.average(numpy.array(line[2:], dtype=numpy.float32))
                            ### we want dissimilarity
                            dis_sims[marker]['all'][(line[0], line[1])] = 1 - sim
                        else:
                            for sub, sub_val in enumerate(line[2:]):
                                if sub not in dis_sims[marker].keys():
                                    dis_sims[marker][sub] = dict()  
                                dis_sims[marker][sub][(line[0], line[1])] = 1 - float(sub_val)
    if args.lang != 'en':
        print(args.lang)
        trans_vocab = set()
        for w in test_vocab:
            try:
                trs_w = trans_from_en[args.lang][w]
            except KeyError:
                print(w)
                continue
            trans_vocab = trans_vocab.union(trs_w)
        del test_vocab
        test_vocab = trans_vocab.copy()

    to_be_removed = list()
    ### we consider a minimum of 7 items following Nili et al. 2014
    threshold = (7*6)+1
    for k in dis_sims.keys():
        if len(dis_sims[k]['all'].keys()) < threshold:
            to_be_removed.append(dis_sims[k]['all'])
    dis_sims = {k : {k_two : v_two for k_two, v_two in v.items()} for k, v in dis_sims.items() if k not in to_be_removed}

    return dis_sims, test_vocab

def read_abstract_ipc(args):
    dis_sims = {'de_abstract-fmri_{}'.format(args.stat_approach) : {'all' : dict()}}
    test_vocab = set()
    file_path = os.path.join(
                             'data', 
                             'fmri',
                             'kaiser',
                             'de_abstract-ipc_data.tsv')
    with open(file_path) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            w_one = line[0]
            w_two = line[1]
            test_vocab = test_vocab.union(transform_german_word(w_one))
            test_vocab = test_vocab.union(transform_german_word(w_two))
            assert len(line[2:]) == 19
            ### the dataset provides already dissimilarities
            dis_sim = numpy.average(numpy.array(line[2:], dtype=numpy.float32))
            dis_sims['de_abstract-fmri_{}'.format(args.stat_approach)]['all'][(w_one, w_two)] = dis_sim

    return dis_sims, test_vocab

def read_mitchell2008(args):
    if args.lang != 'en':
        trans_path = os.path.join(
                                 'data', 
                                 'fmri',
                                 'mitchell',
                                 'mitchell2008_translations_en-de-it.tsv'
                                 )
        trans = dict()
        with open(trans_path) as i:
            for l_i, l in enumerate(i):
                line = l.strip().split('\t')
                if l_i == 0:
                    continue
                if args.lang == 'it':
                    idx = 2
                elif args.lang == 'de':
                    idx = 1
                trans[line[0].strip()] = line[idx].strip()
    test_vocab = set()
    dis_sims = dict()
    for area in ['semantic-network', 'L_IFG', 'L_MTG', 'L_pIPL', 'L_AG', 'L_occipital']:
        key = '{}_mitchell2008_{}#{}_all'.format(args.lang, args.stat_approach, area)
        dis_sims[key] = dict()
        file_path = os.path.join(
                                 'data', 
                                 'fmri',
                                 'mitchell',
                                 'mitchell2008_pairwise',
                                 'tsv',
                                'mitchell2008_pairwise_{}.tsv'.format(area))
        with open(file_path) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                w_one = line[0]
                w_two = line[1]
                ### translating if needed
                if args.lang != 'en':
                    w_one = trans[w_one]
                    w_two = trans[w_two]
                ### german will be transformed
                if args.lang != 'de':
                    test_vocab = test_vocab.union(
                                                  set(
                              transform_basic_word(w_one)+transform_basic_word(w_two)
                                                      )
                                                  )
                else:
                    test_vocab = test_vocab.union(transform_german_word(w_one))
                    test_vocab = test_vocab.union(transform_german_word(w_two))
                ### we use dissimilarity!
                for sub, sim in enumerate(line[2:]):
                    if sub not in dis_sims[key].keys():
                        dis_sims[key][sub] = dict()
                    dis_sims[key][sub][(w_one, w_two)] = 1 - float(sim)

    return dis_sims, test_vocab
