import numpy
import os

from utf_utils import transform_german_word

def read_de_sem_phon_tms():
    sims = dict()
    test_vocab = set()
    lines = list()
    na_lines = list()
    with open(os.path.join(
                           'data', 
                           'tms', 
                           'de_sem-phon', 
                           'de_tms_sem-phon_ifg.tsv')
                           ) as i:
        for l_i, l in enumerate(i):
            line = l.replace(',', '.').strip().split('\t')
            if 'NA' in line:
                #print(line)
                if 'sem' in line:
                    na_lines.append(line)
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
    print('sem trials containing a NA: {}'.format(len(na_lines)))
    ###
    conditions = set([l[header.index('stim')] for l in lines])
    tasks = set([l[header.index('task')] for l in lines])
    full_sims = dict()
    #for c, name in conditions.items():
    for t in tasks:
        for c in conditions:
            name = '{}-{}'.format(c, t)
            #print(name)
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
    full_sims = reorganize_tms_sims(sims)

    return full_sims, test_vocab

def read_de_sound_act_tms():
    ### first reading ratings
    ratings = dict()
    with open(os.path.join(
                           'data', 
                           'tms', 
                           'de_sound-act', 
                           'phil_annotated_ratings_v9.tsv')) as i:
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
    with open(os.path.join(
                           'data', 
                           'tms', 
                           'de_sound-act', 
                           'de_tms_pipl.tsv')
                           ) as i:
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
            #print(top)
            if percent == 'fifty':
                print('top-fifty percent prototypes for {}:\n'.format(tenned))
                print(sorted(top))
                print('\n')
            prototypes['{}-top{}'.format(tenned, percent)] = top

    full_sims = dict()
    conditions = set([l[header.index('condition')] for l in lines])
    tasks = set([l[header.index('task')] for l in lines])
    for c in conditions:
        ### both tasks together
        current_cond = [l for l in lines if l[header.index('condition')]==c]
        subjects = [int(l[header.index('subject')]) for l in current_cond]
        log_rts = [numpy.log10(float(l[header.index('log_rt')])) for l in current_cond]
        rts = [float(l[header.index('rt')]) for l in current_cond]
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
            rts = [float(l[header.index('rt')]) for l in current_cond]
            vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('task')])]
            test_vocab = test_vocab.union(set(vocab_w_ones))
            vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('stimulus')])]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            w_ones = [l[header.index('task')] for l in current_cond]
            w_twos = [l[header.index('stimulus')] for l in current_cond]
            sims['{}_{}'.format(t, c)]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
    full_sims = reorganize_tms_sims(sims)

    return full_sims, test_vocab, prototypes

def read_it_distr_learn_tms():
    lines = list()
    with open(os.path.join(
                           'data',
                           'tms',
                           'it_distr-learn',
                           'italian_tms_cereb.tsv')) as i:
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
        for m_i, marker in enumerate(['1', '0', 'all']):
            if m_i < 2:
                current_cond = [l for l in lines if l[header.index('condition')]==name and l[header.index('Meaningful')]==marker]
            else:
                current_cond = [l for l in lines if l[header.index('condition')]==name]
            log_rts = [numpy.log10(float(l[header.index('RTs')].replace(',', '.'))) for l in current_cond]
            rts = [float(l[header.index('RTs')].replace(',', '.')) for l in current_cond]
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
    related_full_sims = reorganize_tms_sims(related_sims)
    unrelated_full_sims = reorganize_tms_sims(unrelated_sims)
    all_full_sims = reorganize_tms_sims(all_sims)
    
    return related_full_sims, unrelated_full_sims, all_full_sims, test_vocab

def reorganize_tms_sims(sims):
    full_sims = dict()
    for n, n_data in sims.items():
        full_sims[n] = dict()
        for s, ws, rt in n_data:
            if s not in full_sims[n].keys():
                full_sims[n][s] = dict()
            full_sims[n][s][ws] = rt
    return full_sims
