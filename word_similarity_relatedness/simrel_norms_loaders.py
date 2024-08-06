
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
    dis_sims = {'norms' : {'all' : dict()}}
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
            dis_sims['norms']['all'][key] = 1 - val
    return dis_sims, test_vocab

def read_men(lang):
    dis_sims = {'norms' : {'all' : dict()}}
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
                ### transforming to dissimilarity
                dis_sims['norms']['all'][key] = 1 - val
    return dis_sims, test_vocab

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
    dis_sims = {'norms' : {'all' : dict()}}
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
            dis_sims['norms']['all'][key] = 1 - val
    return dis_sims, test_vocab
