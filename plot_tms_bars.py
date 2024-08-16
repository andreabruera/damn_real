import matplotlib
import numpy
import os
import random
import re
import scipy

from matplotlib import colormaps, font_manager, pyplot
from scipy import stats

from plot_utils import font_setup

font_folder = '/data/tu_bruera/fonts'
font_setup(font_folder)

results = dict()

for root, direc, fz in os.walk(
                          os.path.join(
                              'test_results',
                              )):

    for f in fz:
        with open(os.path.join(root, f)) as i:
            for l in i:
                if 'rowincol' in l:
                    continue
                line = l.strip().split('\t')
                lang = line[0]
                if lang not in results.keys():
                    results[lang] = dict()
                model = line[1]
                if 'fasttext' not in model and 'mitchell' not in model and 'concept' not in model:
                    if 'lancaster' not in l:
                        continue
                    if 'random' in l:
                        continue
                    #if 'hi-' or 'lo-' in l:
                        continue
                    num = float(model.split('_')[-2])
                    if 'wiki' in model:
                        short_model = '_'.join(model.split('_')[2:-2])
                    else:
                        short_model = '_'.join(model.split('_')[1:-2])
                task = line[2]
                ### modality
                assert task[:3] == '{}_'.format(lang)
                task_setup = task[3:].split('#')[0]
                if 'distr-learn' in task_setup:
                    mode = 'tms'
                elif 'sem-phon' in task_setup:
                    mode = 'tms'
                elif 'sound-act' in task_setup:
                    mode = 'tms'
                elif 'abstract' in task_setup:
                    mode = 'fmri'
                elif 'fern' in task_setup:
                    mode = 'fmri'
                elif 'dirani' in task_setup:
                    mode = 'meeg'
                elif 'word-naming' in task_setup:
                    mode = 'behavioural'
                elif 'lexical-decision' in task_setup:
                    mode = 'behavioural'
                else:
                    mode = 'simrel-norms'
                if mode not in results[lang].keys():
                    results[lang][mode] = dict()
                if task not in results[lang][mode].keys():
                    results[lang][mode][task] = dict()
                non_nan_res = [v if v!='nan' else 0. for v in line[3:]]
                res = numpy.array(non_nan_res, dtype=numpy.float32)
                if 'fasttext' not in model and 'mitchell' not in model and 'concept' not in model:
                    if short_model not in results[lang][mode][task].keys():
                        results[lang][mode][task][short_model] = dict()
                    results[lang][mode][task][short_model][num] = res
                else:
                    results[lang][mode][task][model] = res
'''
import pdb; pdb.set_trace()

### finding the best models
model_results = dict()
for l, l_data in results.items():
    model_results[l] = dict()
    ### only brain tasks
    rel_tasks = [d for d in l_data.keys() if 
                                           #'353' in d or '999' in d 
                                           #or
                                           'fern' in d or 'dira' in d
                                           ]
    #assert len(rel_tasks) == 4
    assert len(rel_tasks) == 4
    for t in rel_tasks:
        c_results = list()
        for model, m_data in l_data[t].items():
            #print([model, task])
            if type(m_data) == dict:
                for num, num_res in m_data.items():
                    key = '{}_{}'.format(model, num)
                    #if 'top' not in key:
                    #    continue
                    if 'lancaster' not in key:
                        continue
                    c_results.append((key, numpy.average(num_res)))
            else:
                c_results.append((model, numpy.average(m_data)))
        sorted_results = sorted(c_results, key=lambda item : item[1], reverse=True)
        for rank, vals in enumerate(sorted_results):
            model = vals[0]
            if model not in model_results[l].keys():
                model_results[l][model] = [rank+1]
                #model_results[l][model] = [vals[1]]
            else:
                model_results[l][model].append(rank+1)
                #model_results[l][model].append(vals[1])

lang_best = dict()
overall_best = dict()
for l, l_data in model_results.items():
    #print(l_data.keys())
    sorted_ranks = sorted({k : numpy.average(v) for k, v in l_data.items()}.items(), key=lambda item : item[1])
    #sorted_ranks = sorted({k : numpy.average(v) for k, v in l_data.items()}.items(), key=lambda item : item[1], reverse=True)
    print(l)
    print(sorted_ranks[:10])
    best_ft = min([r_i for r_i, r in enumerate(sorted_ranks) if 'fasttext' in r[0] and 'concept' not in r[0]])
    best_other = min([r_i for r_i, r in enumerate(sorted_ranks) if 'fasttext' not in r[0] and 'concept' not in r[0]])
    print(l)
    print(sorted_ranks[:20])
    lang_best[l] = (sorted_ranks[best_ft][0], sorted_ranks[best_other][0])
    if l == 'en':
        continue
    for rank, model in enumerate(sorted_ranks):
        if model[0] not in overall_best.keys():
            overall_best[model[0]] = [rank]
        else:
            overall_best[model[0]].append(rank)
overall_sorted_ranks = sorted({k : numpy.average(v) for k, v in overall_best.items()}.items(), key=lambda item : item[1])
print(overall_sorted_ranks[:10])
best_ft = overall_sorted_ranks[min([r_i for r_i, r in enumerate(overall_sorted_ranks) if 'fasttext' in r[0] and 'concept' not in r[0]])][0]
#best_ft = 'conceptnet'
best_other = overall_sorted_ranks[min([r_i for r_i, r in enumerate(overall_sorted_ranks) if 'fasttext' not in r[0] and 'concept' not in r[0]])][0]
'''
best_ft = 'fasttext'
#best_other = 'cc100_lancaster_freq_top__5000.0'
best_other = 'fasttext'
#best_other = 'cc100_lancaster_freq_hi-perceptual__2500.0'
print('using models: {}, {}'.format(best_ft, best_other))

### datasets where the comparisons are very simple
### just clear-cut pairwise comparisons
for mode in [
             'residualize',
             'bootstrap', 
             'simple',
             ]:
    for l, l_data in results.items():
        if l == 'en':
            continue
        out = os.path.join(
                   'tms_barplots',
                   l, 
                   )
        os.makedirs(out, exist_ok=True)
        rel_tasks = [k.split('#')[0] for k in l_data['tms'].keys() if mode in k and 'act' not in k]
        #assert len(rel_tasks) in [1, 2]
        for t in rel_tasks:
            fig, ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10))
            ax.set_ylim(bottom=-0.04, top=0.4)
            curr_ts = sorted([w for w in l_data['tms'].keys() if t in w and mode in w])
            if len(curr_ts) == 2:
                xs = [0.5]
            elif len(curr_ts) == 3:
                xs = []
            elif len(curr_ts) == 6:
                xs = [1.5, 3.5]
            ax.vlines(
                      x=xs,
                      ymin=-.5,
                      ymax=.5,
                      linewidth=10,
                      linestyles='dashed',
                      color='black',
                      )
            xs = [w.split('#')[1] for w in curr_ts]
            ax.bar(0, 0, color='mediumaquamarine', label='fasttext')
            ax.bar(0, 0, color='goldenrod', label=best_other)
            ### results for fasttext
            #ft_model = lang_best[l][0]
            ft_model = best_ft
            ys = [l_data['tms'][c_t][ft_model] for c_t in curr_ts]
            print(xs)
            print(len(ys))
            print([len(y) for y in ys])
            ax.bar(
                   [x-0.15 for x in range(len(xs))],
                   [numpy.average(y) for y in ys],
                   color='mediumaquamarine',
                   width=0.25,
                   )
            ax.scatter(
                   [x-0.15+(random.randint(-100, 100)*0.001) for x in range(len(xs)) for y in ys[x]],
                   ys,
                   edgecolor='aquamarine',
                   color='white',
                   alpha=0.2,
                   zorder=3.,
                   )
            ### p-values
            for y_i, y in enumerate(ys):
                for y_two_i, y_two in enumerate(ys):
                    if y_two_i <= y_i:
                        continue
                    if len(ys) == 6:
                        if y_i % 2 != 0:
                            continue
                        if y_two_i>y_i+1:
                            continue
                        y_corr = 0.008
                    else:
                        y_corr = y_two_i*0.01
                    #print(y)
                    #print(y_two)
                    p = scipy.stats.wilcoxon(x=y, y=y_two).pvalue
                    #p = scipy.stats.ttest_ind(y, y_two).pvalue
                    if p < 0.99:
                        ax.text(
                                x=y_i-.2,
                                y=-y_corr,
                                s='{} - p={}'.format(xs[y_two_i], round(p, 3)),
                                color='mediumaquamarine',
                                )
            ### results for other model
            other_model = best_other
            #first_part = '_'.join(other_model.split('_')[:-1])
            #second_part = float(other_model.split('_')[-1])
            #ys = [l_data[c_t][first_part][second_part] for c_t in curr_ts]
            ys = [l_data['tms'][c_t]['fasttext'] for c_t in curr_ts]
            ax.bar(
                   [x+0.15 for x in range(len(xs))],
                   [numpy.average(y) for y in ys],
                   color='goldenrod',
                   width=0.25,
                   )
            ax.scatter(
                   [x+0.15+(random.randint(-100, 100)*0.001) for x in range(len(xs)) for y in ys[x]],
                   ys,
                   edgecolor='goldenrod',
                   color='white',
                   alpha=0.2,
                   zorder=3.,
                   )
            ### p-values
            for y_i, y in enumerate(ys):
                for y_two_i, y_two in enumerate(ys):
                    if y_two_i <= y_i:
                        continue
                    if len(ys) == 6:
                        if y_i % 2 != 0:
                            continue
                        if y_two_i>y_i+1:
                            continue
                        y_corr = 0.016
                    else:
                        y_corr = y_two_i*0.01
                    #print(y)
                    #print(y_two)
                    if numpy.average(y) > numpy.average(y_two):
                        alternative='greater'
                        used_x = y_i
                    elif numpy.average(y) < numpy.average(y_two):
                        alternative='less'
                        used_x = y_two_i
                    p = scipy.stats.wilcoxon(x=y, y=y_two, alternative=alternative).pvalue
                    #p = scipy.stats.wilcoxon(x=y, y=y_two).pvalue
                    #p = scipy.stats.ttest_ind(y, y_two).pvalue
                    if p < 0.99:
                        ax.text(
                                x=y_i+.2,
                                y=-y_corr,
                                s='{} - p={}'.format(xs[y_two_i], round(p, 3)),
                                color='goldenrod',
                                )
            ax.legend(fontsize=20,
                      ncols=2,
                      loc=9,
                      )
            pyplot.ylabel(
                          ylabel='log(RT) - semantic dissimilarity correlation',
                          fontsize=23,
                          fontweight='bold'
                          )
            pyplot.yticks(fontsize=20)
            pyplot.xticks(
                         ticks=range(len(xs)),
                         labels=[x.replace('_', ' ').replace('-', '\n') for x in xs],
                         fontsize=25,
                         fontweight='bold'
                         )
            pyplot.title(
                         '{}'.format(t.replace('_', ' ')), 
                         fontsize=30,
                         )
            pyplot.savefig(
                    os.path.join(
                        out, 
                       '{}.jpg'.format(t)))
            pyplot.clf()
            pyplot.close()
    ### phil's dataset, where it is not clear how to look at results
    significance = 0.05
    for model in [
                  best_ft, 
                  #best_other,
                  ]:
        for l, l_data in results.items():
            if l != 'de':
                continue
            out = os.path.join(
                       'tms_barplots',
                       l, 
                       )
            os.makedirs(out, exist_ok=True)
            all_rel_tasks = [k.split('#')[0] for k in l_data['tms'].keys() if mode in k and 'act' in k]

            print(set(all_rel_tasks))
            assert len(set(all_rel_tasks)) == 1
            #for t in all_rel_tasks:
            soundact= list(set((all_rel_tasks)))[0]
            indiv_bars = sorted(set([w.split('#')[-1].split('_')[0] for w in l_data['tms'].keys() if soundact in w and mode in w]))
            #print(cases)
            #for c in cases:
            #    #indiv_bars = sorted(['_'.join(w.split('#')[-1].split('_')[1:]) for w in l_data['tms'].keys() if t in w and curr_t])
            #    #indiv_bars = sorted(set([t.split('_')[0] for t in curr_ts]))
            #    indiv_bars = sorted(set([w.split('#')[-1].split('_')[0] for w in l_data['tms'].keys() if soundact in w and c in w and mode in w]))
            #    print(indiv_bars)
            corrections = {b : v for b, v in zip(indiv_bars, numpy.linspace(-.35, .35, len(indiv_bars)))}
            #print(corrections)
            fig, ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10))
            ax.scatter(
                       x=0., 
                       y=1.,
                       marker='*',
                       s=100,
                       color='black',
                       #alpha=0.,
                       label='p<{}'.format(significance)
                       )
            ax.set_ylim(bottom=-0.26, top=0.26)
            xs = [1.5, 3.5]
            #xticks = sorted(set([w.split('#')[1].split('_')[1] for w in l_data['tms'].keys() if soundact in w and mode in w]))
            #print(xticks)
            ax.vlines(
                      x=xs,
                      ymin=-.2,
                      ymax=.24,
                      linewidth=5,
                      linestyles='dashed',
                      color='black',
                      )
            ax.vlines(
                      x=[0.5, 2.5, 4.5],
                      ymin=-.2,
                      ymax=.24,
                      linewidth=3,
                      linestyles='dashed',
                      color='silver',
                      )
            #xs = [w.split('_tms_')[1] for w in curr_ts]
            colors_l = [
                    #'paleturquoise', 
                    #'mediumaquamarine', 
                    'forestgreen', 
                    #'mediumblue',
                    #'palegoldenrod', 
                    'palevioletred',
                    'silver',
                    'orange', 
                    'teal',
                    'magenta',
                    'black',
                    'indianred',
                    'slateblue',
                    'hotpink',
                    ]
            colors = dict()
            for k, col in zip(corrections.keys(), colors_l):
                ax.bar(0, 0, color=col, label=k)
                colors[k] = col
                #ax.bar(0, 0, color='goldenrod', label=best_other)
            ### results for fasttext
            #ft_model = lang_best[l][0]
            x_ticks = list()
            for k, corr in corrections.items():
                #curr_ts = sorted([w for w in l_data.keys() if t in w and k in w])
                curr_ts = sorted([w for w in l_data['tms'].keys() if k in w and soundact in w and mode in w])
                if len(x_ticks) == 0:
                    x_ticks = ['\n'.join(w.split('#')[-1].split('_')[1:]) for w in curr_ts]
                else:
                    new_x_ticks = ['\n'.join(w.split('#')[-1].split('_')[1:]) for w in curr_ts]
                    assert new_x_ticks == x_ticks
                if 'fasttext' in model:
                    ys = [l_data['tms'][c_t][model] for c_t in curr_ts]
                else:
                    first_part = '_'.join(model.split('_')[:-1])
                    second_part = float(model.split('_')[-1])
                    ys = [l_data[c_t][first_part][second_part] for c_t in curr_ts]
                width=0.8/len(ys)
                ax.bar(
                       [x+corr for x in range(len(ys))],
                       [numpy.average(y) for y in ys],
                       #color='mediumaquamarine',
                       width=width,
                       color=colors[k]
                       )
                ax.scatter(
                       [x+((random.randint(-8, 8)*0.1)/len(ys))+corr for x in range(len(ys)) for y in ys[x]],
                       ys,
                       edgecolor=colors[k],
                       color='white',
                       alpha=0.1,
                       zorder=3.,
                       )
                ### p-values
                for y_i, y in enumerate(ys):
                    for y_two_i, y_two in enumerate(ys):
                        if y_two_i <= y_i:
                            continue
                        if len(ys) == 6:
                            if y_i % 2 != 0:
                                continue
                            if y_two_i>y_i+1:
                                continue
                        #print(y)
                        #print(y_two)
                        ### setting the directionality
                        if numpy.average(y) > numpy.average(y_two):
                            alternative='greater'
                            used_x = y_i
                        elif numpy.average(y) < numpy.average(y_two):
                            alternative='less'
                            used_x = y_two_i

                        #alternative = 
                        p = scipy.stats.wilcoxon(x=y, y=y_two, 
                                                 #alternative=alternative,
                                                 ).pvalue
                        #print(p)
                        #p = scipy.stats.ttest_ind(y, y_two).pvalue
                        if p < significance:
                            #ax.text(
                            #        x=y_i-.2+corr,
                            #        y=-y_corr,
                            #        s='{} - p={}'.format(curr_ts[y_two_i], round(p, 3)),
                            #        color='mediumaquamarine',
                            #        )
                            ax.scatter(
                                       x=used_x+corr, 
                                       y=-.2,
                                       marker='*',
                                       s=100,
                                       color=colors[k]
                                       )

            ax.legend(fontsize=15,
                      ncols=7,
                      loc=8,
                      )
            pyplot.ylabel(
                          ylabel='log(RT) - semantic dissimilarity correlation',
                          fontsize=23,
                          fontweight='bold'
                          )
            pyplot.yticks(fontsize=20)
            pyplot.xticks(
                         ticks=range(len(x_ticks)),
                         labels=[x.replace('Geraeusch', 'sound-task').replace('Handlung', 'action-task') for x in x_ticks],
                         fontsize=25,
                         fontweight='bold'
                         )
            pyplot.title(
                         '{} - {}'.format(t.replace('_', ' '), model), 
                         fontsize=30,
                         )
            pyplot.savefig(
                    os.path.join(
                        out, 
                       '{}-{}-{}.jpg'.format(mode, soundact, model)))
            pyplot.clf()
            pyplot.close()
