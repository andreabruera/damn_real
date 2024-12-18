import numpy
import os

from tqdm import tqdm
from utf_utils import transform_german_word

def read_italian_anew(args):
    ### lexical decition times
    sims = dict()
    test_vocab = set()
    for task in [
                  'lexical-decision', 
                  'word-naming', 
                  ]:
        full_case = 'it_anew-{}_{}#all-words'.format(task, args.stat_approach)
        sims[full_case] = dict()
        with open(os.path.join('data', 'behavioural', 'it_anew', 'it_anew-{}.tsv'.format(task))) as i:
            for l_i, l in enumerate(i):
                line = l.replace(',', '.').split('\t')
                #print(line)
                if l_i == 0:
                    header = [w.strip() for w in line]
                assert len(line) == len(header)
                if task == 'lexical-decision':
                    if line[header.index('Stimulus_Type')] != 'Word':
                        continue
                if line[header.index('Accuracy')] != '1':
                    continue
                word = line[header.index('Ita_Word')].lower()
                sub = line[header.index('Subject')]
                ### compressing...
                sub = 'all'
                if sub not in sims[full_case].keys():
                    sims[full_case][sub] = dict()
                test_vocab = test_vocab.union(set([word, word.capitalize()]))
                try:
                    word_rt = float(line[header.index('RTs')])
                except ValueError:
                    print([sub, word])
                    continue
                try:
                    sims[full_case][sub][word].append(word_rt)
                except KeyError:
                    sims[full_case][sub][word] = [word_rt]
    final_sims = dict()
    with tqdm() as cntr:
        for case, case_r in sims.items():
            final_sims[case] = dict()
            for sub, measures in case_r.items():
                final_sims[case][sub] = list()
                for k_one_i, k_one in enumerate(sorted(measures.keys())):
                    for k_two_i, k_two in enumerate(sorted(measures.keys())):
                        if k_two_i <= k_one_i:
                            continue
                        key = tuple(sorted([k_one, k_two]))
                        final_sims[case][sub].append((key, abs(numpy.average(measures[k_one])-numpy.average(measures[k_two]))))
                        cntr.update(1)
    return final_sims, test_vocab

def read_italian_blindsighted(args):
    ### lexical decition times
    #sims = {'it_lexical-decision-blindsighted_{}'.format(args.stat_approach) : dict()}
    sims = dict()
    test_vocab = set()
    for sub_type in ['SC', 'EB']:
        for w_typ in [
                      'abs', 
                      'vis', 
                      'multi',
                      'all'
                      ]:
            full_case = 'it_lexical-decision-blindsighted_{}#{}_{}-words'.format(sub_type, args.stat_approach, w_typ)
            sims[full_case] = dict()
            with open(os.path.join('data', 'behavioural', 'it_lexical-decision-blindsighted.tsv')) as i:
                for l_i, l in enumerate(i):
                    line = l.replace(',', '.').split('\t')
                    #print(line)
                    if l_i == 0:
                        header = [w.strip() for w in line]
                    assert len(line) == len(header)
                    if line[header.index('Condition')] != 'word':
                        continue
                    if line[header.index('Group')] != sub_type:
                        continue
                    if line[header.index('accuracy')] != '1':
                        continue
                    if w_typ != 'all':
                        if line[header.index('Type')] != w_typ:
                            continue
                    word = line[header.index('Stimulus')].lower()
                    sub = line[header.index('SubjectID')]
                    if sub not in sims[full_case].keys():
                        sims[full_case][sub] = dict()
                    test_vocab = test_vocab.union(set([word, word.capitalize()]))
                    try:
                        word_rt = float(line[header.index('RT')])
                    except ValueError:
                        print([sub, word])
                        continue
                    sims[full_case][sub][word] = word_rt
    final_sims = dict()
    for case, case_r in sims.items():
        final_sims[case] = dict()
        for sub, measures in case_r.items():
            final_sims[case][sub] = list()
            for k_one_i, k_one in enumerate(sorted(measures.keys())):
                for k_two_i, k_two in enumerate(sorted(measures.keys())):
                    if k_two_i <= k_one_i:
                        continue
                    key = tuple(sorted([k_one, k_two]))
                    final_sims[case][sub].append((key, abs(measures[k_one]-measures[k_two])))
    return final_sims, test_vocab

def read_italian_deafhearing(args):
    ### lexical decition times
    #sims = {'it_lexical-decision-deafhearing_{}'.format(args.stat_approach) : dict()}
    sims = dict()
    test_vocab = set()
    for sub in ['Hearing', 'LIS', 'SP']:
        case = 'it_lexical-decision-deafhearing_{}#{}'.format(args.stat_approach, sub)
        sims[case] = dict()
        short_case = case.split('_')[1]
        with open(os.path.join('data', 'behavioural', 'it_lexical-decision-deafhearing.tsv'.format(short_case))) as i:
            for l_i, l in enumerate(i):
                line = l.replace(',', '.').split('\t')
                if l_i == 0:
                    first_header = [w.strip() for w in line]
                    subs = [h_i for h_i, h in enumerate(first_header) if sub in h]
                    print(subs)
                    continue
                if l_i == 1:
                    second_header = [w.strip() for w in line]
                    continue
                assert len(first_header) == len(second_header)
                assert len(line) == len(first_header)
                if line[second_header.index('Lexicality')] != 'Word':
                    continue
                word = line[second_header.index('Stimuli')].lower()
                for sub in subs:
                    if sub not in sims[case].keys():
                        sims[case][sub] = dict()
                    test_vocab = test_vocab.union(set([word, word.capitalize()]))
                    try:
                        words_rt = numpy.log10(float(line[sub]))
                    except ValueError:
                        #print(line[sub])
                        assert line[sub] in ['\n',  '']
                        #print([sub, word])
                        continue
                    sims[case][sub][word] = words_rt
    final_sims = dict()
    for case, case_r in sims.items():
        final_sims[case] = dict()
        for sub, measures in case_r.items():
            final_sims[case][sub] = list()
            for k_one_i, k_one in enumerate(sorted(measures.keys())):
                for k_two_i, k_two in enumerate(sorted(measures.keys())):
                    if k_two_i <= k_one_i:
                        continue
                    key = tuple(sorted([k_one, k_two]))
                    final_sims[case][sub].append((key, abs(measures[k_one]-measures[k_two])))
    return final_sims, test_vocab

def read_italian_mouse(args):
    ### lexical decition times
    sims = {'it_abs-conc-decision_{}'.format(args.stat_approach) : dict()}
    test_vocab = set()
    for case in sims.keys(): 
        short_case = case.split('_')[1]
        measures = dict()
        with open(os.path.join('data', 'behavioural', 'abs-conc-mouse_it', 'abs-conc-mouse_it.tsv'.format(short_case))) as i:
            for l_i, l in enumerate(i):
                line = l.replace(',', '.').strip().split('\t')
                if l_i == 0:
                    header = [w for w in line]
                    continue
                word = line[header.index('text')].lower()
                word_two = line[header.index('text2')].lower()
                sub = line[header.index('ID')]
                if sub not in sims[case].keys():
                    sims[case][sub] = dict()
                test_vocab = test_vocab.union(set([word, word.capitalize(), word_two, word_two.capitalize()]))
                words_rt = float(line[header.index('RTs')])
                sims[case][sub][(word, word_two)] = words_rt
    return sims, test_vocab

def read_picnaming_seven(args):
    if args.lang == 'en':
        w_key = 'ename'
    elif args.lang ==  'it':
        w_key = 'iname'
    elif args.lang == 'de':
        w_key = 'gname'
    ### lexical picture naming times in seven languages
    sims = {'{}_picture-naming-seven_{}'.format(args.lang, args.stat_approach) : {'all' : dict()}}
    test_vocab = set()
    for case in sims.keys(): 
        short_case = case.split('_')[1]
        measures = dict()
        with open(os.path.join('data', 'behavioural', 'picture-naming_seven-languages.tsv')) as i:
            for l_i, l in enumerate(i):
                line = [w.strip() for w in l.replace(',', '.').split('\t')]
                if l_i == 0:
                    header = [w for w in line]
                    continue
                word = line[header.index(w_key)].lower()
                if args.lang != 'de':
                    test_vocab = test_vocab.union(set([word, word.capitalize()]))
                else:
                    test_vocab = test_vocab.union(set(transform_german_word(word)))
                word_rt = float(line[header.index('erttar')])
                measures[word] = word_rt
        for k_one_i, k_one in enumerate(sorted(measures.keys())):
            for k_two_i, k_two in enumerate(sorted(measures.keys())):
                if k_two_i <= k_one_i:
                    continue
                key = tuple(sorted([k_one, k_two]))
                sims[case]['all'][key] = abs(measures[k_one]-measures[k_two])
    return sims, test_vocab

def read_italian_behav(args):
    ### lexical decition times
    sims = {'it_word-naming_{}'.format(args.stat_approach) : {'all' : dict()}}
    test_vocab = set()
    for case in sims.keys(): 
        short_case = case.split('_')[1]
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

def read_german_behav(args):
    ### lexical decition times
    sims = {'de_word-naming_{}'.format(args.stat_approach) : {'all' : list()}, 'de_lexical-decision_{}'.format(args.stat_approach) : {'all' : list()}}
    test_vocab = set()
    for case in sims.keys(): 
        short_case = case.split('_')[1]
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
                sims[case]['all'].append((key, abs(measures[k_one]-measures[k_two])))
    return sims, test_vocab
