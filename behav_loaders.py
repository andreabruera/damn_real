import os

from utf_utils import transform_german_word

def read_italian_behav():
    ### lexical decition times
    sims = {'it_word-naming' : {'all' : dict()}}
    test_vocab = set()
    for case in sims.keys(): 
        short_case = case.replace('it_', '')
        measures = dict()
        with open(os.path.join('data', 'behavioural', 'LexVar', 'lexvar_{}_it.tsv'.format(short_case))) as i:
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
                sims[case]['all'][key] = abs(measures[k_one]-measures[k_two])
    return sims, test_vocab

def read_german_behav():
    ### lexical decition times
    sims = {'de_word-naming' : {'all' : dict()}, 'de_lexical-decision' : {'all' : dict()}}
    test_vocab = set()
    for case in sims.keys(): 
        short_case = case.replace('de_', '')
        measures = dict()
        with open(os.path.join('data', 'behavioural', 'DeveL', 'devel_{}_de.tsv'.format(short_case))) as i:
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
                sims[case]['all'][key] = abs(measures[k_one]-measures[k_two])
    return sims, test_vocab
