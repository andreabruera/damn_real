import matplotlib
import numpy
import os
import re

from matplotlib import colormaps, pyplot

results = dict()

with open('evaluation.tsv') as i:
    for l in i:
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
        res = float(line[3])
        if 'fasttext' not in model and 'mitchell' not in model:
            if short_model not in results[lang][task].keys():
                results[lang][task][short_model] = dict()
            results[lang][task][short_model][num] = res
        else:
            results[lang][task][model] = res

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
                      y=t_res[ft],
                      xmin=-.1,
                      xmax=max(all_vals)+.1,
                      label=ft,
                      linestyles=style,
                      color='red',
                      )
        for mitch, col in zip(mitchs, numpy.linspace(0, 1, len(mitchs))):
            ax.hlines(
                      y=t_res[mitch],
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
                    ys_freqs,
                    label=case,
                    color=colors[case]
                    )
            ax.scatter(
                    xs_freqs,
                    ys_freqs,
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
