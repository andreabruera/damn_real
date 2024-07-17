import argparse
import matplotlib
import numpy
import os
import random
import scipy
import sys

from matplotlib import pyplot
from utf_utils import transform_german_word

def read_italian_cereb_tms(lang):
    lines = list()
    with open(os.path.join('tms', 'data', 'italian_tms_cereb.tsv')) as i:
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
        if lang == 'it':
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
                    related_sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, rts)]
                elif m_i == 1:
                    unrelated_sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, rts)]
                elif m_i == 2:
                    all_sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, rts)]
        else:
            related_full_sims[name] = list()
            unrelated_full_sims[name] = list()
            all_full_sims[name] = list()
    if lang == 'it':
        related_full_sims = reorganize_tms_sims(related_sims)
        unrelated_full_sims = reorganize_tms_sims(unrelated_sims)
        all_full_sims = reorganize_tms_sims(all_sims)
    return related_full_sims, unrelated_full_sims, all_full_sims, test_vocab

def read_german_pipl_tms(lang):
    ### first reading ratings
    ratings = dict()
    with open(os.path.join('tms', 'data', 'phil_annotated_ratings_v9.tsv')) as i:
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
    with open(os.path.join('tms', 'data', 'de_tms_pipl.tsv')) as i:
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
            print(top)
            prototypes['{}-top{}'.format(tenned, percent)] = top

    full_sims = dict()
    conditions = set([l[header.index('condition')] for l in lines])
    tasks = set([l[header.index('task')] for l in lines])
    for c in conditions:
        if lang == 'de':
            ### both tasks together
            current_cond = [l for l in lines if l[header.index('condition')]==c]
            subjects = [int(l[header.index('subject')]) for l in current_cond]
            log_rts = [numpy.log10(float(l[header.index('rt')])) for l in current_cond]
            rts = [float(l[header.index('rt')]) for l in current_cond]
            vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('task')])]
            test_vocab = test_vocab.union(set(vocab_w_ones))
            vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('stimulus')])]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            w_ones = [l[header.index('task')] for l in current_cond]
            w_twos = [l[header.index('stimulus')] for l in current_cond]
            sims[c]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, rts)]
            for t in tasks:
                ### separate tasks
                current_cond = [l for l in lines if l[header.index('condition')]==c and l[header.index('task')]==t]
                subjects = [int(l[header.index('subject')]) for l in current_cond]
                rts = [float(l[header.index('rt')]) for l in current_cond]
                log_rts = [numpy.log10(float(l[header.index('rt')])) for l in current_cond]
                exps = [l[header.index('expected_response')] for l in current_cond]
                #log_rts = [float(l[header.index('rt')]) for l in current_cond]
                vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('task')])]
                test_vocab = test_vocab.union(set(vocab_w_ones))
                vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('stimulus')])]
                test_vocab = test_vocab.union(set(vocab_w_twos))
                w_ones = [l[header.index('task')] for l in current_cond]
                w_twos = [l[header.index('stimulus')] for l in current_cond]
                sims['{}_{}'.format(t, c)]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, rts)]
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
    #with open(os.path.join('tms', 'data', 'german_ifg_r.tsv')) as i:
    with open(os.path.join('tms', 'data', 'original', 'de_tms_sem-phon_ifg.tsv')) as i:
        for l_i, l in enumerate(i):
            line = l.replace(',', '.').strip().split('\t')
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
    ###
    conditions = set([l[header.index('stim')] for l in lines])
    tasks = set([l[header.index('task')] for l in lines])
    full_sims = dict()
    #for c, name in conditions.items():
    for t in tasks:
        for c in conditions:
            name = '{}-{}'.format(c, t)
            #print(name)
            if lang == 'de':
                ###One participant was replaced due to an overall mean error rate of 41.8% - sub 3
                #current_cond = [l for l in lines if l[header.index('stim')] in c and int(l[header.index('subj')])!=3]
                current_cond = [l for l in lines if l[header.index('stim')] in name and l[header.index('task')] in name and int(l[header.index('subj')])!=3]
                tasks = [l[header.index('task')] for l in current_cond]
                assert len(set(tasks)) == 1
                subjects = [int(l[header.index('subj')]) for l in current_cond]
                assert len(set(subjects)) == 24
                #print(subjects)
                #log_rts = [float(l[header.index('log_rt')]) for l in current_cond]
                rts = [float(l[header.index('RT')]) for l in current_cond]
                log_rts = [numpy.log10(float(l[header.index('RT')])) for l in current_cond]
                vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('utterance')])]
                test_vocab = test_vocab.union(set(vocab_w_ones))
                vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('item')].split('.')[0])]
                test_vocab = test_vocab.union(set(vocab_w_twos))
                w_ones = [l[header.index('utterance')] for l in current_cond]
                w_twos = [l[header.index('item')].split('.')[0] for l in current_cond]
                sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, rts)]
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

germ_tms_ifg, germ_tms_ifg_vocab = read_german_ifg_tms('de')
## german TMS
overall_min = min([v for n_data in germ_tms_ifg.values() for s_data in n_data.values() for v in s_data.values()])
overall_max = max([v for n_data in germ_tms_ifg.values() for s_data in n_data.values() for v in s_data.values()])
overall_mean = numpy.average([v for n_data in germ_tms_ifg.values() for s_data in n_data.values() for v in s_data.values()])
overall_std = numpy.std([v for n_data in germ_tms_ifg.values() for s_data in n_data.values() for v in s_data.values()])
dataset = 'de_sem-phon'
counter = 0
colors = ['teal', 'orange', 'violet']
out = os.path.join('tms', 'tms_raw_plots')
os.makedirs(out, exist_ok=True)
fig, ax = pyplot.subplots(constrained_layout=True)
ax.set_ylim(bottom=1400, top=2300)
#ax.set_ylim(bottom=3., top=3.5)
xticks = list()
err_counter = 0
for case, results in [
    ('aIFG', germ_tms_ifg['aIFG-sem'],),
    ('pIFG', germ_tms_ifg['pIFG-sem'], ),
    ('vertex', germ_tms_ifg['vertex-sem'],),
    ]:
    #ys = [numpy.average([(val-overall_min)/(overall_max-overall_min) for val in s_data.values()]) for s_data in results.values()]
    ys = [numpy.average([val for val in s_data.values()]) for s_data in results.values()]
    ### removing 
    ### Naming latencies deviating from a participant's mean (aggregated by task and stimulation
    ### site) by more than three SDs were treated as outliers and removed
    ### from the analysis (59 observations, 0.8% of all trials)
    ys = list()
    for s, s_data in results.items():
        s_ys = list()
        s_data_mean = numpy.nanmean([v for res in germ_tms_ifg.values() for v in res[s].values()])
        s_data_std = numpy.nanstd([v for res in germ_tms_ifg.values() for v in res[s].values()])
        max_threshold = s_data_mean+(s_data_std*3)
        min_threshold = s_data_mean-(s_data_std*3)
        for val in s_data.values():
            '''
            ### removing overly slow/fast responses
            if val > max_threshold:
                err_counter += 1
                continue
            if val < min_threshold:
                err_counter += 1
                continue
            '''
            s_ys.append(val)
        ys.append(numpy.average(s_ys))
        #ys.extend(s_ys)
    #print(ys)
    #ys = [val for s_data in results.values() for val in s_data.values()]
    #ax.bar(counter, numpy.median(ys), color=colors[counter])
    ax.bar(counter, numpy.average(ys), color=colors[counter])
    ax.scatter(
           [counter+random.randint(-100, 100)*0.001 for y in range(len(ys))], 
           ys,
           color='white',
           edgecolors = colors[counter]
           )
    counter += 1
    xticks.append(case)
print(err_counter)
pyplot.xticks(
              ticks=range(len(xticks)),
              labels=xticks,
              rotation=45,
              )
pyplot.ylabel('RT')
pyplot.title('basic RTs for {}'.format(dataset))
pyplot.savefig(os.path.join(out,  '{}.jpg'.format(dataset)))
pyplot.clf()
pyplot.close()
### sound-act
de_tms_pipl, de_tms_pipl_vocab, prototypes = read_german_pipl_tms('de')
overall_min = min([v for n_data in de_tms_pipl.values() for s_data in n_data.values() for v in s_data.values()])
overall_max = max([v for n_data in de_tms_pipl.values() for s_data in n_data.values() for v in s_data.values()])
overall_mean = numpy.average([v for n_data in de_tms_pipl.values() for s_data in n_data.values() for v in s_data.values()])
overall_std = numpy.std([v for n_data in de_tms_pipl.values() for s_data in n_data.values() for v in s_data.values()])
colors = ['gainsboro', 'goldenrod', 'mediumseagreen', 'coral', 'dodgerblue', 'orchid']
dataset = 'de_sound-act'
xticks=list()
fig, ax = pyplot.subplots(constrained_layout=True)
#ax.set_ylim(bottom=0.43,top=0.51)
ax.set_ylim(bottom=500,top=1600)
counter = 0
for case, results in [
            ('actiontask-pIPL', de_tms_pipl['Handlung_pIPL'], ),
            ('actiontask-sham', de_tms_pipl['Handlung_sham'], ),
            ('all-pIPL', de_tms_pipl['pIPL'], ),
            ('all-sham', de_tms_pipl['sham'], ),
            ('soundtask-pIPL', de_tms_pipl['Geraeusch_pIPL'], ),
            ('soundtask-sham', de_tms_pipl['Geraeusch_sham'], ),
            ]:
    #ys = [numpy.average([(val-overall_min)/(overall_max-overall_min) for val in s_data.values()]) for s_data in results.values()]
    ys = [numpy.average([val for val in s_data.values()]) for s_data in results.values()]
    #ys = [val for s_data in results.values() for val in s_data.values()]
    ax.bar(counter, numpy.average(ys), color=colors[counter])
    ax.scatter(
           [counter+random.randint(-100, 100)*0.001 for y in range(len(ys))], 
           ys,
           color='white',
           edgecolors = colors[counter]
           )
    counter += 1
    xticks.append(case)
pyplot.xticks(
              ticks=range(len(xticks)),
              labels=xticks,
              rotation=45,
              )
pyplot.ylabel('RTs')
pyplot.title('basic RTs for {}'.format(dataset))
pyplot.savefig(os.path.join(out,  '{}.jpg'.format(dataset)))
pyplot.clf()
pyplot.close()
related_ita_tms_cereb, unrelated_ita_tms_cereb, all_ita_tms_cereb, ita_tms_cereb_vocab = read_italian_cereb_tms('it')
overall_min = min([v for n_data in all_ita_tms_cereb.values() for s_data in n_data.values() for v in s_data.values()])
overall_max = max([v for n_data in all_ita_tms_cereb.values() for s_data in n_data.values() for v in s_data.values()])
colors = ['gainsboro', 'goldenrod', 'mediumseagreen', 'coral', 'dodgerblue', 'orchid']
dataset = 'it_distr-learn'
xticks=list()
fig, ax = pyplot.subplots(constrained_layout=True)
#ax.set_ylim(bottom=2., top=3.5)
counter = 0
for case, results in [
                    ## italian TMS
                    ('all_cereb', all_ita_tms_cereb['cedx'], ),
                    ('all_vertex', all_ita_tms_cereb['cz'], ),
                    ('related_cereb', related_ita_tms_cereb['cedx'], ),
                    ('related_vertex', related_ita_tms_cereb['cz'], ),
                    ('unrelated_cereb', unrelated_ita_tms_cereb['cedx'], ),
                    ('unrelated_vertex', unrelated_ita_tms_cereb['cz'], ),
                    ]:
    #ys = [numpy.average([(val-overall_min)/(overall_max-overall_min) for val in s_data.values()]) for s_data in results.values()]
    ys = [numpy.average([val for val in s_data.values()]) for s_data in results.values()]
    #ys = [val for s_data in results.values() for val in s_data.values()]
    ax.bar(counter, numpy.average(ys), color=colors[counter])
    ax.scatter(
           [counter+random.randint(-100, 100)*0.001 for y in range(len(ys))], 
           ys,
           color='white',
           edgecolors = colors[counter]
           )
    counter += 1
    xticks.append(case)
pyplot.xticks(
              ticks=range(len(xticks)),
              labels=xticks,
              rotation=45,
              )
pyplot.ylabel('RTs')
pyplot.title('basic RTs for {}'.format(dataset))
pyplot.savefig(os.path.join(out,  '{}.jpg'.format(dataset)))
pyplot.clf()
pyplot.close()
