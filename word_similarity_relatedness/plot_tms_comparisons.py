import matplotlib
import numpy
import os
import random
import re

from matplotlib import colormaps, pyplot

results = dict()

with open('evaluation.tsv') as i:
    for l in i:
        if 'rowincol' in l:
            continue
        if 'perceptual' in l:
            continue
        line = l.strip().split('\t')
        lang = line[0]
        if lang not in results.keys():
            results[lang] = dict()
        model = line[1]
        if 'fasttext' not in model and 'mitchell' not in model:
            num = float(model.split('_')[-2])
            if 'wiki' in model:
                short_model = '_'.join(model.split('_')[2:-2])
            else:
                short_model = '_'.join(model.split('_')[1:-2])
        task = line[2]
        if task not in results[lang].keys():
            results[lang][task] = dict()
        res = numpy.array(line[3:], dtype=numpy.float32)
        if 'fasttext' not in model and 'mitchell' not in model:
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
            else:
                model_results[l][model].append(rank+1)

lang_best = dict()
for l, l_data in model_results.items():
    #print(l_data.keys())
    sorted_ranks = sorted({k : numpy.average(v) for k, v in l_data.items()}.items(), key=lambda item : item[1])
    #print(l)
    best_ft = min([r_i for r_i, r in enumerate(sorted_ranks) if 'fasttext' in r[0]])
    best_other = min([r_i for r_i, r in enumerate(sorted_ranks) if 'fasttext' not in r[0]])
    lang_best[l] = (sorted_ranks[best_ft][0], sorted_ranks[best_other][0])

for l, l_data in results.items():
    if l == 'en':
        continue
    rel_tasks = set([d.split('_tms_')[0] for d in l_data.keys() if 'tms' in d])
    #print(rel_tasks)
    assert len(rel_tasks) in [1, 2]
    for t in rel_tasks:
        fig, ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10))
        ax.set_ylim(bottom=0., top=0.3)
        curr_ts = [w for w in l_data.keys() if t in w]
        xs = [w.split('_tms_')[1] for w in l_data.keys() if t in w]
        ax.bar(0, 0, color='mediumaquamarine', label='fasttext')
        ax.bar(0, 0, color='goldenrod', label='')
        ### results for fasttext
        ft_model = lang_best[l][0]
        ys = [l_data[c_t][ft_model] for c_t in curr_ts]
        ax.bar(
               [x-0.25 for x in range(len(xs))],
               [numpy.average(y) for y in ys],
               color='mediumaquamarine',
               )
        ### results for other model
        other_model = lang_best[l][1]
        first_part = '_'.join(other_model.split('_')[:-1])
        second_part = float(other_model.split('_')[-1])
        ys = [l_data[c_t][first_part][second_part] for c_t in curr_ts]
        ax.bar(
               [x+0.25 for x in range(len(xs))],
               [numpy.average(y) for y in ys],
               color='mediumaquamarine',
               )
        ax.scatter(
               [x+0.25+(random.randint(-100, 100)*0.001) for x in range(len(xs))],
               [numpy.average(y) for y in ys],
               edgecolor='mediumaquamarine',
               color='white',
               alpha=0.5,
               zorder=3.,
               )
        pyplot.ylabel(
                      ylabel='RT-semantic dissimilarity correlation',
                      fontsize=23,
                      )
        pyplot.yticks(fontsize=20)
        pyplot.xticks(
                     ticks=range(len(xs)),
                     labels=[x.replace('_', ' ') for x in xs],
                     fontsize=25,
                     )
        pyplot.title(
                     '{}'.format(t.replace('_', ' ')), 
                     fontsize=30,
                     )
        pyplot.savefig(os.path.join('sim_rel', l, '{}.jpg'.format(t)))
        pyplot.clf()
        pyplot.close()


import pdb; pdb.set_trace()

for lang, l_res in results.items():
    folder = os.path.join('sim_rel', lang)
    os.makedirs(folder, exist_ok=True)
    for task, t_res in l_res.items():
        fig, ax = pyplot.subplots(
                                  constrained_layout=True,
                                  figsize=(20, 10),
                                  )
        if 'men' in task or '353' in task:
            top = 0.85
        elif 'simlex' in task:
            top = 0.5
        elif 'tms' in task:
            top = 0.35
        else:
            top = 0.2
        bottom = -.08 if task != 'men' else 0.1
        ax.set_ylim(bottom=bottom, top=top)
        all_vals = list()
        fts = [k for k in t_res.keys() if 'fast' in k]
        mitchs = [k for k in t_res.keys() if 'mitch' in k and 'rowincol' not in k]
        #others = {k : sorted(vals.items(), key=lambda item : item[0]) for k, vals in t_res.items() if 'fast' not in k and 'mitch' not in k}
        others = {k : sorted(vals.items(), key=lambda item : item[0]) for k, vals in t_res.items() if 'fast' not in k and 'mitch' not in k and 'top' in k and 'rowincol' not in k}
        all_vals = [val[0] for v in others.values() for val in v] + [0.]
        ax.hlines(
                  y=0,
                  xmin=-.1,
                  xmax=max(all_vals)+.1,
                  color='black',
                  )
        for ft, style in zip(fts, ['solid', 'dashdot',]):
            ax.hlines(
                      y=numpy.average(t_res[ft]),
                      xmin=-.1,
                      xmax=max(all_vals)+.1,
                      label=ft,
                      linestyles=style,
                      color='red',
                      )
        for mitch, col in zip(mitchs, numpy.linspace(0, 1, len(mitchs))):
            ax.hlines(
                      y=numpy.average(t_res[mitch]),
                      xmin=-.1,
                      xmax=max(all_vals)+.1,
                      label=mitch,
                      color=matplotlib.colormaps['cividis'](col)
                      )
        colors = {k : v for k, v in zip(others.keys(), matplotlib.cm.rainbow(numpy.linspace(0, 1, len(others.keys()))))}
        for case, sort_freqs in others.items():
            xs_freqs = [v[0] for v in sort_freqs]
            ys_freqs = [v[1] for v in sort_freqs]
            ax.plot(
                    xs_freqs,
                    [numpy.average(v) for v in ys_freqs],
                    label=case,
                    color=colors[case]
                    )
            ax.scatter(
                    xs_freqs,
                    [numpy.average(v) for v in ys_freqs],
                    s=50,
                    marker='d',
                    edgecolors='white',
                    linewidths=1,
                    color=colors[case],
                    zorder=3
                    )
        pyplot.legend(
                ncols=4,
                loc=8,
                fontsize=10,
                )
        pyplot.savefig(
                        os.path.join(
                                     folder,
                             '{}.jpg'.format(task))
                        )
        pyplot.clf()
        pyplot.close()
