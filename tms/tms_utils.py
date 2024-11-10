import numpy
import os

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
