import numpy
import os

def read_fern(lang, trans_from_en):
    dis_sims = {1 : {'all' : dict()}, 2 : dict()}
    test_vocab = set()
    for dataset in dis_sims.keys():
        dis_sims[dataset] = {'all' : dict()}
        file_path = os.path.join(
                                 'data', 
                                 'fmri', 
                                 'fernandino', 
                                 'fern{}_semantic_network.tsv'.format(dataset)
                                 )
        with open(file_path) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                test_vocab = test_vocab.union(set([line[0], line[1]]))
                if dataset == 1:
                    sim = numpy.average(numpy.array(line[2:], dtype=numpy.float32))
                    ### we want dissimilarity
                    dis_sims[dataset]['all'][(line[0], line[1])] = 1 - sim
                else:
                    for sub, sub_val in enumerate(line[2:]):
                        if sub not in dis_sims[dataset].keys():
                            dis_sims[dataset][sub] = dict()  
                        dis_sims[dataset][sub][(line[0], line[1])] = 1 - float(sub_val)
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

    return dis_sims, test_vocab

def read_fern_categories(lang, trans_from_en):
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
    for dataset in [1, 2]:
        file_path = os.path.join(
                                 'data', 
                                 'fmri', 
                                 'fernandino', 
                                 'fern{}_semantic_network.tsv'.format(dataset)
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
                test_vocab = test_vocab.union(set([line[0], line[1]]))
                sim = numpy.average(numpy.array(line[2:], dtype=numpy.float32))
                ### abstract/concrete: idx 0
                ### finer grained categories: idx 1
                for idx in [
                            0, 
                            #1,
                            ]:
                    cat = set([mapper[line[0]][idx], mapper[line[1]][idx]])
                    if len(cat) == 1:
                        marker = '{}-{}'.format(dataset, list(cat)[0])
                        if marker not in dis_sims.keys():
                            dis_sims[marker] = {'all' : dict()}
                        ### we want dissimilarity
                        dis_sims[marker]['all'][(line[0], line[1])] = 1 - sim
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

    to_be_removed = list()
    for k in dis_sims.keys():
        if len(dis_sims[k]['all'].keys()) < 100:
            to_be_removed.append(dis_sims[k]['all'])
    dis_sims = {k : {'all' : v['all']} for k, v in dis_sims.items() if k not in to_be_removed}

    return dis_sims, test_vocab

def read_abstract_ipc(lang):
    dis_sims = {'ipc' : {'all' : dict()}}
    test_vocab = set()
    if lang == 'de':
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
                dis_sims['ipc']['all'][(w_one, w_two)] = dis_sim

    return dis_sims, test_vocab
