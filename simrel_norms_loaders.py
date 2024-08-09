import os

from utf_utils import transform_german_word

def read_ws353(lang):
    base_folder = os.path.join('data', 'simrel_norms', 'ws353')
    if lang == 'de':
        file_path = os.path.join(base_folder, 'MWS353_German.txt')
    if lang == 'it':
        file_path = os.path.join(base_folder, 'MWS353_Italian.txt')
    if lang == 'en':
        file_path = os.path.join(base_folder, 'MWS353_English.txt')
    indices = [0, 1, -1]
    sep = ','
    assert os.path.exists(file_path)
    case = '{}_ws353'.format(args.lang)
    dis_sims = {case : {'all' : dict()}}
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
            ### transforming to dissimilarity
            dis_sims[case]['all'][key] = 1 - val
    return dis_sims, test_vocab

def read_men():
    dis_sims = {'en_men' : {'all' : dict()}}
    test_vocab = set()
    file_path = os.path.join('data', 'simrel_norms', 'men', 'MEN_dataset_natural_form_full')
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
            ### transforming to dissimilarity
            dis_sims['en_men']['all'][key] = 1 - val
    return dis_sims, test_vocab

def read_simlex(lang):
    base_folder = os.path.join('data', 'simrel_norms', 'simlex999')
    if lang == 'de':
        file_path = os.path.join(base_folder, 'MSimLex999_German.txt')
        indices = [0, 1, -1]
        sep = ','
    if lang == 'it':
        file_path = os.path.join(base_folder, 'MSimLex999_Italian.txt')
        indices = [0, 1, -1]
        sep = ','
    if lang == 'en':
        file_path = os.path.join(base_folder, 'SimLex-999.txt')
        indices = [0, 1, 3]
        sep = '\t'
    assert os.path.exists(file_path)
    case = '{}_simlex999'.format(args.lang)
    dis_sims = {case : {'all' : dict()}}
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
            ### transforming to dissimilarity
            dis_sims[case]['all'][key] = 1 - val
    return dis_sims, test_vocab
