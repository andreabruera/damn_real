import matplotlib
import numpy
import os
import random
import re
import scipy

from matplotlib import colormaps, font_manager, pyplot
from scipy import stats

### Font setup
# Using Helvetica as a font
font_folder = '/import/cogsci/andrea/dataset/fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)

results = dict()

with open('evaluation.tsv') as i:
    for l in i:
        if 'rowincol' in l:
            continue
        #if 'perceptual' in l:
        #    continue
        line = l.strip().split('\t')
        lang = line[0]
        if lang not in results.keys():
            results[lang] = dict()
        model = line[1]
        if 'fasttext' not in model and 'mitchell' not in model and 'concept' not in model:
            num = float(model.split('_')[-2])
            if 'wiki' in model:
                short_model = '_'.join(model.split('_')[2:-2])
            else:
                short_model = '_'.join(model.split('_')[1:-2])
        task = line[2]
        if task not in results[lang].keys():
            results[lang][task] = dict()
        res = numpy.array(line[3:], dtype=numpy.float32)
        if 'fasttext' not in model and 'mitchell' not in model and 'concept' not in model:
            if short_model not in results[lang][task].keys():
                results[lang][task][short_model] = dict()
            results[lang][task][short_model][num] = res
        else:
            results[lang][task][model] = res

### finding the best models
model_results = dict()
for l, l_data in results.items():
    model_results[l] = dict()
    rel_tasks = [d for d in l_data.keys() if '353' in d or '999' in d]
    assert len(rel_tasks) == 2
    for t in rel_tasks:
        c_results = list()
        for model, m_data in l_data[t].items():
            #print([model, task])
            if type(m_data) == dict:
                for num, num_res in m_data.items():
                    key = '{}_{}'.format(model, num)
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
best_ft = 'conceptnet'
best_other = overall_sorted_ranks[min([r_i for r_i, r in enumerate(overall_sorted_ranks) if 'fasttext' not in r[0] and 'concept' not in r[0]])][0]

for l, l_data in results.items():
    if l == 'en':
        continue
    rel_tasks = set([d.split('_tms_')[0] for d in l_data.keys() if 'tms' in d])
    print(rel_tasks)
    assert len(rel_tasks) in [1, 2]
    for t in rel_tasks:
        fig, ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10))
        ax.set_ylim(bottom=-0.04, top=0.3)
        curr_ts = sorted([w for w in l_data.keys() if t in w])
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
        xs = [w.split('_tms_')[1] for w in curr_ts]
        ax.bar(0, 0, color='mediumaquamarine', label='fasttext')
        ax.bar(0, 0, color='goldenrod', label=best_other)
        ### results for fasttext
        #ft_model = lang_best[l][0]
        ft_model = best_ft
        ys = [l_data[c_t][ft_model] for c_t in curr_ts]
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
               alpha=0.5,
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
        first_part = '_'.join(other_model.split('_')[:-1])
        second_part = float(other_model.split('_')[-1])
        ys = [l_data[c_t][first_part][second_part] for c_t in curr_ts]
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
               alpha=0.5,
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
                p = scipy.stats.wilcoxon(x=y, y=y_two).pvalue
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
                      ylabel='RT - semantic dissimilarity correlation',
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
        pyplot.savefig(os.path.join('sim_rel', l, '{}.jpg'.format(t)))
        pyplot.clf()
        pyplot.close()
