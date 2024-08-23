import numpy
import os

from utf_utils import transform_german_word

def read_de_pmtg_production_tms(args):
    lines = list()
    with open(os.path.join(
                           'data',
                           'tms',
                           'de_pmtg-production.tsv')) as i:
        for l_i, l in enumerate(i):
            line = l.split('\t')
            if l_i == 0:
                header = [w.strip() for w in line]
                continue
            lines.append([w.strip() for w in line])
    stims = set([l[header.index('stimulation')] for l in lines])
    conds = {
             'u' : 'unrelated',
             'r' : 'related',
             'ur' : 'all-but-same',
             'urt' : 'all',
             }
    all_sims = dict()
    test_vocab = set()
    for name, cond in conds.items():
        for stim in stims:
            print(name)
            key = 'de_pmtg-production_{}#{}-{}'.format(args.stat_approach, cond, stim)
            current_cond = [l for l in lines if l[header.index('condition')].strip() in name and \
                                                l[header.index('stimulation')] == stim and \
                                                l[header.index('response')] not in ['0', 'NA'] and \
                                                l[header.index('rt')] not in ['0', 'NA']
                                                ]
            log_rts = [numpy.log10(float(l[header.index('rt')])) for l in current_cond]
            rts = [float(l[header.index('rt')]) for l in current_cond]
            subjects = [int(l[header.index('sbj')]) for l in current_cond]
            vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('picture')].split('.')[0])]
            test_vocab = test_vocab.union(set(vocab_w_ones))
            vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('distractor')])]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            w_ones = [l[header.index('picture')].split('.')[0] for l in current_cond]
            w_twos = [l[header.index('distractor')].strip() for l in current_cond]
            all_sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
    final_sims = reorganize_tms_sims(all_sims)

    return final_sims, test_vocab

def read_de_sem_phon_tms(args):
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
            ### removing trailing spaces
            line = [w.strip() for w in line]
            lines.append(line)
    print('sem trials containing a NA: {}'.format(len(na_lines)))
    ###
    conditions = set([l[header.index('stim')] for l in lines])
    tasks = set([l[header.index('task')] for l in lines])
    #print(tasks)
    full_sims = dict()
    #for c, name in conditions.items():
    for t in tasks:
        if 'sem' not in t:
            continue
        for c in conditions:
            name = 'de_sem-phon_{}#{}-{}'.format(args.stat_approach, t, c)
            #print(name)
            ###One participant was replaced due to an overall mean error rate of 41.8% - sub 3
            #current_cond = [l for l in lines if l[header.index('stim')] in c and int(l[header.index('subj')])!=3]
            current_cond = [l for l in lines if l[header.index('stim')] in name and l[header.index('task')] in t and int(l[header.index('subj')])!=3]
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
    collect_info(full_sims)

    return full_sims, test_vocab

def read_it_distr_learn_tms(args):
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

    final_sims = {'it_distr-learn_{}#all-trials_{}'.format(args.stat_approach, k) : v for k, v in all_full_sims.items()}
    for k, v in related_full_sims.items():
        final_sims['it_distr-learn_{}#related-trials_{}'.format(args.stat_approach, k)] = v
    for k, v in unrelated_full_sims.items():
        final_sims['it_distr-learn_{}#unrelated-trials_{}'.format(args.stat_approach, k)] = v
    collect_info(final_sims)
    
    return final_sims, test_vocab

def reorganize_tms_sims(sims):
    full_sims = dict()
    for n, n_data in sims.items():
        full_sims[n] = dict()
        for s, ws, rt in n_data:
            if s not in full_sims[n].keys():
                full_sims[n][s] = dict()
            full_sims[n][s][ws] = rt
    return full_sims

def read_phil_ratings():
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
    return ratings

def read_soundact_prototypes(ratings):
    ### possibilities in task-modelling:
    # centroid overall (all)
    # both positive (both_pos)
    # both negative (both_neg)
    # matched exclusive (action_pos_sound_neg, sound_pos_action_neg)
    # matched non-exclusive (action_pos, sound_pos)
    prototypes = {
                  'action_pos-all' : list(), 
                  'sound_pos-all' : list(),
                  'action_neg-all' : list(),
                  'sound_neg-all' : list(), 
                  'all-all-all' : list(),
                  'all-pos-all' : list(),
                  'all-neg-all' : list(),
                  'action_pos_sound_neg-all' : list(), 
                  'sound_pos_action_neg-all' : list(),
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
            ### checking words are all there
            if line[header.index('sound_word')] != 'NA':
                if w not in ratings.keys():
                    missing.append(w)
            if line[header.index('action_word')] != 'NA':
                if w not in ratings.keys():
                    missing.append(w)
            assert len(missing) == 0
            ### distributing prototypes
            ### both
            if line[header.index('action_word')] == '-1' and line[header.index('sound_word')] == '-1':
                prototypes['all-neg-all'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '1' and line[header.index('sound_word')] == '1':
                prototypes['all-pos-all'].append(line[header.index('stimulus')])
            ### exclusive
            if line[header.index('action_word')] == '1' and line[header.index('sound_word')] == '-1':
                prototypes['action_pos_sound_neg-all'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '-1' and line[header.index('sound_word')] == '1':
                prototypes['sound_pos_action_neg-all'].append(line[header.index('stimulus')])
            ### inclusive
            if line[header.index('action_word')] == '1':
                prototypes['action_pos-all'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '-1':
                prototypes['action_neg-all'].append(line[header.index('stimulus')])
            if line[header.index('sound_word')] == '1':
                prototypes['sound_pos-all'].append(line[header.index('stimulus')])
            if line[header.index('sound_word')] == '-1':
                prototypes['sound_neg-all'].append(line[header.index('stimulus')])
            ### if it isn't lexical decision, drop it
            if 'lexical_decision' in line:
                continue
            ### everything
            prototypes['all-all-all'].append(line[header.index('stimulus')])
    prototypes = {k : set(v) for k, v in prototypes.items()}
    #for k, v in prototypes.items():
    #    print('\n')
    #    print('prototypes for {}'.format(k))
    #    print(v)
    ### using only 0.1, 0.5 highest rated words
    top_tenned = [
          'action_pos-all', 
          'sound_pos-all',
          'all-pos-all',
          'all-neg-all',
          'action_pos_sound_neg-all', 
          'sound_pos_action_neg-all',
          ]
    for tenned in top_tenned:
        ### top ten percent
        for percent, mult in [('ten', 0.1), ('fifty', 0.5)]:
            ten_percent = int(len(prototypes[tenned])*mult)
            ### only sound
            if 'sound_pos' in tenned:
                top = [s[0] for s in sorted([(k, ratings[k]['sound']) for k in prototypes[tenned]], key=lambda item : item[1], reverse=True)][:ten_percent]
            ### only action
            if 'action_pos' in tenned:
                top = [s[0] for s in sorted([(k, ratings[k]['action']) for k in prototypes[tenned]], key=lambda item : item[1], reverse=True)][:ten_percent]
            ### both
            else:
                top_s = [s[0] for s in sorted([(k, ratings[k]['sound']) for k in prototypes[tenned]], key=lambda item : item[1], reverse=True)]
                top_a = [s[0] for s in sorted([(k, ratings[k]['action']) for k in prototypes[tenned]], key=lambda item : item[1], reverse=True)]
                top = [s[0] for s in sorted([(k, top_a.index(k)+top_s.index(k)) for k in prototypes[tenned]], key=lambda item : item[1])][:ten_percent]
            #if percent == 'fifty':
            #    print('top-fifty percent prototypes for {}:\n'.format(tenned))
            #    print(sorted(top))
            #    print('\n')
            prototypes['{}-top{}'.format(tenned[:-4], percent)] = top
    prototypes = {k : tuple(v) if type(v)!=tuple else v for k, v in prototypes.items()}
    return prototypes

def return_proto_words(task, proto_mode, prototypes):
    val = proto_mode.split('-')[-1]
    if proto_mode in [
                      'all-all-all', 
                      'all-pos-all', 
                      'all-neg-all',
                      'all-pos-topten', 
                      'all-pos-topfifty', 
                      ]:
        words = prototypes['{}'.format(proto_mode)]
    else:
        if 'incl' in proto_mode:
            if 'matched' in proto_mode:
                ### sound
                if task == 'Geraeusch':
                    words = prototypes['sound_pos-{}'.format(val)]
                elif task == 'Handlung':
                    words = prototypes['action_pos-{}'.format(val)]
                else:
                    raise RuntimeError()
            elif 'opposite' in proto_mode:
                ### sound
                if task == 'Geraeusch':
                    words = prototypes['sound_neg-{}'.format(val)]
                elif task == 'Handlung':
                    words = prototypes['action_neg-{}'.format(val)]
                else:
                    raise RuntimeError()
            else:
                raise RuntimeError()
        elif 'excl' in proto_mode:
            if 'matched' in proto_mode:
                ### sound
                if task == 'Geraeusch':
                    words = prototypes['sound_pos_action_neg-{}'.format(val)]
                elif task == 'Handlung':
                    words = prototypes['action_pos_sound_neg-{}'.format(val)]
                else:
                    raise RuntimeError()
            elif 'opposite' in proto_mode:
                ### sound
                if task == 'Geraeusch':
                    words = prototypes['action_pos_sound_neg-{}'.format(val)]
                elif task == 'Handlung':
                    words = prototypes['sound_pos_action_neg-{}'.format(val)]
                else:
                    raise RuntimeError()
            else:
                raise RuntimeError()
        else:
            raise RuntimeError()
    return words

def read_de_sound_act_tms(args):
    ratings = read_phil_ratings()
    prototypes = read_soundact_prototypes(ratings)
    ### reading dataset
    lines = list()
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
            if 'lexical_decision' in line:
                continue
            lines.append(line)

    proto_modes = [
                 #'all-all-all', 
                 'all-pos-all',
                 #'all-pos-topten',
                 #'all-pos-topfifty',
                 'all-neg-all',
                 'matched-excl-all',
                 #'matched-excl-topten',
                 #'matched-excl-topfifty',
                 #'matched-incl-all',
                 #'matched-incl-topten',
                 #'matched-incl-topfifty',
                 'opposite-excl-all',
                 #'opposite-excl-topten',
                 #'opposite-excl-topfifty',
                 #'opposite-incl-all',
                 #'opposite-incl-topten',
                 #'opposite-incl-topfifty',
                 ]
    sims = dict()
    test_vocab = set()
    conditions = set([l[header.index('condition')] for l in lines])
    tasks = set([l[header.index('task')] for l in lines])
    ### everything together
    for proto_mode in proto_modes:
        for c in conditions:
            key = 'de_sound-act-aggregated_{}#{}_all_{}'.format(args.stat_approach, proto_mode, c)
            ### both tasks together
            current_cond = [l for l in lines if l[header.index('condition')]==c]
            subjects = [int(l[header.index('subject')]) for l in current_cond]
            rts = [float(l[header.index('rt')]) for l in current_cond]
            log_rts = [numpy.log10(float(l[header.index('log_rt')])) for l in current_cond]
            ### with prototyping, we actually use words as first items
            #w_ones = [l[header.index('task')] for l in current_cond]
            w_ones = [return_proto_words(l[header.index('task')], proto_mode, prototypes) for l in current_cond]
            all_w_ones = [transform_german_word(w) for ws in w_ones for w in ws]
            test_vocab = test_vocab.union(set([w for ws in all_w_ones for w in ws]))
            ### these are the words subjects actually saw
            vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('stimulus')])]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            w_twos = [l[header.index('stimulus')] for l in current_cond]
            sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
            for t in tasks:
                print('\n')
                print('prototypes for {}, {}'.format(proto_mode, t))
                print(return_proto_words(t, proto_mode, prototypes))
                key = 'de_sound-act-aggregated_{}#{}_{}_{}'.format(args.stat_approach, proto_mode, t, c)
                ### separate tasks
                current_cond = [l for l in lines if l[header.index('condition')]==c and l[header.index('task')]==t]
                subjects = [int(l[header.index('subject')]) for l in current_cond]
                log_rts = [numpy.log10(float(l[header.index('log_rt')])) for l in current_cond]
                exps = [l[header.index('expected_response')] for l in current_cond]
                rts = [float(l[header.index('rt')]) for l in current_cond]
                #vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('task')])]
                #test_vocab = test_vocab.union(set(vocab_w_ones))
                ### with prototyping, we actually use words as first items
                #w_ones = [l[header.index('task')] for l in current_cond]
                w_ones = [return_proto_words(l[header.index('task')], proto_mode, prototypes) for l in current_cond]
                all_w_ones = [transform_german_word(w) for ws in w_ones for w in ws]
                test_vocab = test_vocab.union(set([w for ws in all_w_ones for w in ws]))
                vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('stimulus')])]
                test_vocab = test_vocab.union(set(vocab_w_twos))
                w_twos = [l[header.index('stimulus')] for l in current_cond]
                #sims['{}_{}'.format(t, c)]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
                sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
    ### if it's not fasttext, we just use aggregate
    if args.model == 'fasttext':
        ### everything split
        for action, action_n in [('lo-', '-1'), ('hi-', '1')]:
            for sound, sound_n in [('lo-', '-1'), ('hi-', '1')]:
                for proto_mode in proto_modes:
                    for c in conditions:
                        key = 'de_sound-act-detailed_{}#{}_all-{}sound-{}action_{}'.format(args.stat_approach, proto_mode, sound, action, c)
                        ### both tasks together
                        current_cond = [l for l in lines if l[header.index('condition')]==c and l[header.index('action_word')]==action_n and l[header.index('sound_word')]==sound_n]
                        subjects = [int(l[header.index('subject')]) for l in current_cond]
                        rts = [float(l[header.index('rt')]) for l in current_cond]
                        log_rts = [numpy.log10(float(l[header.index('log_rt')])) for l in current_cond]
                        ### with prototyping, we actually use words as first items
                        #w_ones = [l[header.index('task')] for l in current_cond]
                        w_ones = [return_proto_words(l[header.index('task')], proto_mode, prototypes) for l in current_cond]
                        all_w_ones = [transform_german_word(w) for ws in w_ones for w in ws]
                        test_vocab = test_vocab.union(set([w for ws in all_w_ones for w in ws]))
                        ### these are the words subjects actually saw
                        vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('stimulus')])]
                        test_vocab = test_vocab.union(set(vocab_w_twos))
                        w_twos = [l[header.index('stimulus')] for l in current_cond]
                        sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
                        for t in tasks:
                            print('\n')
                            print('prototypes for {}, {}'.format(proto_mode, t))
                            print(return_proto_words(t, proto_mode, prototypes))
                            key = 'de_sound-act-detailed_{}#{}_{}-{}sound-{}action_{}'.format(args.stat_approach, proto_mode, t, sound, action, c)
                            ### separate tasks
                            current_cond = [l for l in lines if l[header.index('condition')]==c and l[header.index('task')]==t and l[header.index('action_word')]==action_n and l[header.index('sound_word')]==sound_n]
                            subjects = [int(l[header.index('subject')]) for l in current_cond]
                            log_rts = [numpy.log10(float(l[header.index('log_rt')])) for l in current_cond]
                            exps = [l[header.index('expected_response')] for l in current_cond]
                            rts = [float(l[header.index('rt')]) for l in current_cond]
                            #vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('task')])]
                            #test_vocab = test_vocab.union(set(vocab_w_ones))
                            ### with prototyping, we actually use words as first items
                            #w_ones = [l[header.index('task')] for l in current_cond]
                            w_ones = [return_proto_words(l[header.index('task')], proto_mode, prototypes) for l in current_cond]
                            all_w_ones = [transform_german_word(w) for ws in w_ones for w in ws]
                            test_vocab = test_vocab.union(set([w for ws in all_w_ones for w in ws]))
                            vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('stimulus')])]
                            test_vocab = test_vocab.union(set(vocab_w_twos))
                            w_twos = [l[header.index('stimulus')] for l in current_cond]
                            #sims['{}_{}'.format(t, c)]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
                            sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
    full_sims = reorganize_tms_sims(sims)
    collect_info(full_sims)

    return full_sims, test_vocab, prototypes

def collect_info(full_sims):
    labels = set(full_sims.keys())
    subjects = set([s for subs in full_sims.values() for s in subs.keys()])
    trials = set([len(s.keys()) for subs in full_sims.values() for s in subs.values()])
    print('labels: ')
    print(labels)
    print('subjects: ')
    print(subjects)
    print('trials: ')
    print(trials)
