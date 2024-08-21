import matplotlib
import numpy
import os
import re

from matplotlib import colormaps, pyplot
from tqdm import tqdm

results = dict()

with tqdm() as counter:
    for root, direc, fz in os.walk(
                              os.path.join(
                                  'test_results',
                                  #'old_results',
                                  )):

        for f in fz:
            with open(os.path.join(root, f)) as i:
                for l in i:
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
                    dataset = line[2]
                    #print(dataset)
                    if 'en_men' in dataset:
                        task = 'simrel_norms'
                    elif '999' in dataset:
                        task = 'simrel_norms'
                    elif '353' in dataset:
                        task = 'simrel_norms'
                    elif 'fern' in dataset:
                        task = 'fmri'
                    elif 'dirani' in dataset:
                        task = 'meeg'
                    elif 'abstract' in dataset:
                        task = 'fmri'
                    elif 'de_behav' in dataset:
                        task = 'behavioural'
                    elif 'it_behav' in dataset:
                        task = 'behavioural'
                    elif 'sem-phon' in dataset:
                        task = 'tms'
                    elif 'sound-act' in dataset:
                        task = 'tms'
                    elif 'distr-learn' in dataset:
                        task = 'tms'
                    else:
                        print(dataset)
                        #continue
                        raise AssertionError
                    '''
                    if 'sem-' in dataset:
                        task = 'tms'
                    elif 'fern' in dataset or 'abstract-ipc' in dataset:
                        task = 'fmri'
                    elif dataset in ['1-abstract', '2-abstract', '1-concrete', '2-concrete']:
                        task = 'fmri'
                    elif 'dirani' in dataset:
                        task = 'meg'
                    elif 'lexical' in dataset or 'naming' in dataset:
                        task = 'behavioural'
                    elif '999' in dataset or '353' in dataset or 'men' in dataset:
                        task = 'sim-rel_norms'
                    else:
                        print(dataset)
                        continue
                        #raise AssertionError
                    '''
                    if task not in results[lang].keys():
                        results[lang][task] = {dataset : dict()}
                    if dataset not in results[lang][task].keys():
                        results[lang][task][dataset] = dict()
                    res = numpy.array(line[3:], dtype=numpy.float32)
                    if 'fasttext' not in model and 'mitchell' not in model and 'concept' not in model:
                        if short_model not in results[lang][task][dataset].keys():
                            results[lang][task][dataset][short_model] = dict()
                        results[lang][task][dataset][short_model][num] = res
                    else:
                        results[lang][task][dataset][model] = res
                    counter.update(1)

with tqdm() as counter:
    for lang, l_res in results.items():
        for general_task, task_res in l_res.items():
            for task, t_res in task_res.items():
                fig, ax = pyplot.subplots(
                                          constrained_layout=True,
                                          figsize=(20, 10),
                                          )
                ### dividing into lines and regular values
                fts = [k for k in t_res.keys() if 'fast' in k or 'concept' in k]
                mitchs = [k for k in t_res.keys() if 'mitch' in k and 'rowincol' not in k]
                others = {
                          k : sorted(vals.items(), key=lambda item : item[0]) \
                                  for k, vals in t_res.items() \
                                  if 'fast' not in k and \
                                  'mitch' not in k and \
                                  'rowincol' not in k and \
                                  'concept' not in k and \
                                  #('top' in k
                                  (
                                  #'top' in k
                                  #or '_hi-' in k or '_lo-' in k
                                  #or 
                                  'abs' in k and 'random' not in k
                                  #and 'abs' not in k
                                  )
                                  }
                ### plotting horizontal lines
                all_vals = [val[0] for v in others.values() for val in v] + [0.]
                ax.hlines(
                          y=0,
                          xmin=-.1,
                          xmax=max(all_vals)+.1,
                          color='black',
                          )
                alls = list()
                ### fasttext
                for ft, style in zip(fts, ['solid', 'dashdot', 'dotted']):
                    y = numpy.average(t_res[ft])
                    alls.append(y)
                    ax.hlines(
                              y=y,
                              xmin=-.1,
                              xmax=max(all_vals)+.1,
                              label=ft,
                              linestyles=style,
                              color='red',
                              )
                ### mitchell dimensions
                for mitch, col in zip(mitchs, numpy.linspace(0, 1, len(mitchs))):
                    y = numpy.average(t_res[mitch])
                    alls.append(y)
                    ax.hlines(
                              y=y,
                              xmin=-.1,
                              xmax=max(all_vals)+.1,
                              label=mitch,
                              color=matplotlib.colormaps['cividis'](col)
                              )
                ### count models
                ### we use rainbow as a set of colours to sample from
                colors = {k : v for k, v in zip(others.keys(), matplotlib.cm.rainbow(numpy.linspace(0, 1, len(others.keys()))))}
                for case, sort_freqs in others.items():
                    xs_freqs = [v[0] for v in sort_freqs]
                    ys_freqs = [v[1] for v in sort_freqs]
                    ys = [numpy.average(v) for v in ys_freqs]
                    alls.extend(ys)
                    assert len(ys) == len(xs_freqs)
                    ax.plot(
                            xs_freqs,
                            ys,
                            label=case,
                            color=colors[case]
                            )
                    ax.scatter(
                            xs_freqs,
                            ys,
                            s=50,
                            marker='d',
                            edgecolors='white',
                            linewidths=1,
                            color=colors[case],
                            zorder=3
                            )
                ### checking the plot is actually not empty
                if len(alls) == 0:
                    continue
                ### setting plot limits
                corr = numpy.nanstd(alls)
                ax.set_ylim(
                            #bottom=min(0, min(alls))-corr, 
                            #top=max(0, max(alls))+corr
                            bottom=-.2,
                            top=.2,
                            )
                ### legend
                pyplot.legend(
                        ncols=5,
                        loc=8,
                        fontsize=10,
                        )
                ### creating folder
                specific_task = task.split('_')[min(1, len(task.split('_'))-1)]
                assert len(specific_task) > 0
                folder = os.path.join(
                                      'test_lineplots',
                                      lang, 
                                      general_task,
                                      specific_task,
                                      )
                os.makedirs(folder, exist_ok=True)
                ### saving figure
                pyplot.savefig(
                                os.path.join(
                                             folder,
                                     '{}.jpg'.format(task))
                                )
                pyplot.clf()
                pyplot.close()
                counter.update(1)
