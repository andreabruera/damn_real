import os

from utf_utils import transform_german_word

def transform_basic_word(w):
    ws = [w, w.capitalize()]
    return ws

def read_ws353(args):
    base_folder = os.path.join('data', 'simrel_norms', 'ws353')
    if args.lang == 'de':
        file_path = os.path.join(base_folder, 'MWS353_German.txt')
    if args.lang == 'it':
        file_path = os.path.join(base_folder, 'MWS353_Italian.txt')
    if args.lang == 'en':
        file_path = os.path.join(base_folder, 'MWS353_English.txt')
    indices = [0, 1, -1]
    sep = ','
    assert os.path.exists(file_path)
    case = '{}_ws353_{}'.format(args.lang, args.stat_approach)
    dis_sims = {case : {'all' : dict()}}
    test_vocab = set()
    with open(file_path) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.lower().strip().split(sep)
            key = tuple(sorted([line[indices[0]], line[indices[1]]]))
            if args.lang == 'de':
                norm_key = set([m for k in key for w in k for m in transform_german_word(k)])
            else:
                norm_key = set(
                    [bw for ws in key for bw in transform_basic_word(ws)]
                    )
            #print(norm_key)
            test_vocab = test_vocab.union(norm_key)
            val = float(line[indices[2]].replace(',', '.'))
            ### transforming to dissimilarity
            dis_sims[case]['all'][key] = 1 - val
    return dis_sims, test_vocab

def read_men(args):
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
            #norm_key = set([m for k in key for w in k for m in transform_german_word(k)])
            norm_key = set(
                [bw for ws in key for bw in transform_basic_word(ws)]
                )
            #print(norm_key)
            test_vocab = test_vocab.union(norm_key)
            val = float(line[2].replace(',', '.'))
            ### transforming to dissimilarity
            dis_sims['en_men_{}'.format(args.stat_approach)]['all'][key] = 1 - val
    return dis_sims, test_vocab

def read_simlex(args):
    base_folder = os.path.join('data', 'simrel_norms', 'simlex999')
    if args.lang == 'de':
        file_path = os.path.join(base_folder, 'MSimLex999_German.txt')
        indices = [0, 1, -1]
        sep = ','
    if args.lang == 'it':
        file_path = os.path.join(base_folder, 'MSimLex999_Italian.txt')
        indices = [0, 1, -1]
        sep = ','
    if args.lang == 'en':
        file_path = os.path.join(base_folder, 'SimLex-999.txt')
        indices = [0, 1, 3]
        sep = '\t'
    assert os.path.exists(file_path)
    case = '{}_simlex999_{}'.format(args.lang, args.stat_approach)
    dis_sims = {case : {'all' : dict()}}
    test_vocab = set()
    with open(file_path) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.lower().strip().split(sep)
            key = tuple(sorted([line[indices[0]], line[indices[1]]]))
            #norm_key = set([m for k in key for w in k for m in transform_german_word(k)])
            if args.lang == 'de':
                norm_key = set([m for k in key for w in k for m in transform_german_word(k)])
            else:
                norm_key = set(
                [bw for ws in key for bw in transform_basic_word(ws)]
                )
            #print(norm_key)
            test_vocab = test_vocab.union(norm_key)
            val = float(line[indices[2]].replace(',', '.'))
            ### transforming to dissimilarity
            dis_sims[case]['all'][key] = 1 - val
    return dis_sims, test_vocab
