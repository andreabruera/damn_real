import matplotlib
import numpy
import os
import random
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
            #if 'raw' in root: 
            #    continue
            #if 'sym' in root: 
            #    continue
            if 'lm' in root:
                pass
            elif 'llama' in root:
                pass
            else:
                continue
            #print(root)
            with open(os.path.join(root, f)) as i:
                for l in i:
                    line = l.strip().split('\t')
                    lang = line[0]
                    if lang not in results.keys():
                        results[lang] = dict()
                    model = line[1]
                    num = int(model.split('-')[-1])
                    short_model = model.split('_')[0]
                    dataset = line[2]
                    if 'en_men' in dataset:
                        task = 'simrel_norms'
                    elif '999' in dataset:
                        task = 'simrel_norms'
                    elif '353' in dataset:
                        task = 'simrel_norms'
                    elif 'mitchell' in dataset:
                        task = 'fmri'
                    elif 'fern' in dataset:
                        task = 'fmri'
                    elif 'dirani' in dataset:
                        task = 'meeg'
                    elif 'kaneshiro' in dataset:
                        task = 'meeg'
                    elif 'abstract' in dataset:
                        task = 'fmri'
                    elif 'lexical' in dataset:
                        task = 'behavioural'
                    elif 'naming' in dataset:
                        task = 'behavioural'
                    elif 'abs-conc-decision' in dataset:
                        task = 'behavioural'
                    elif 'sem-phon' in dataset:
                        task = 'tms'
                    elif 'sound-act' in dataset:
                        task = 'tms'
                    elif 'pmtg-prod' in dataset:
                        task = 'tms'
                    elif 'distr-learn' in dataset:
                        task = 'tms'
                    elif 'social-quant' in dataset:
                        task = 'tms'
                    else:
                        #continue
                        raise RuntimeError
                    if task not in results[lang].keys():
                        results[lang][task] = {dataset : dict()}
                    if dataset not in results[lang][task].keys():
                        results[lang][task][dataset] = dict()
                    res = numpy.array(line[3:], dtype=numpy.float32)
                    if short_model not in results[lang][task][dataset].keys():
                        results[lang][task][dataset][short_model] = dict()
                    results[lang][task][dataset][short_model][num] = res
                    counter.update(1)

with tqdm() as counter:
    for lang, l_res in results.items():
        print(lang)
        for general_task, task_res in l_res.items():
            print(general_task)
            for task, t_res in task_res.items():
                print(task)
                ### creating folder
                specific_task = task.split('_')[min(1, len(task.split('_'))-1)]
                assert len(specific_task) > 0
                folder = os.path.join(
                                      'test_llms_lineplots',
                                      lang, 
                                      general_task,
                                      specific_task,
                                      )
                if not os.path.exists(folder):
                    os.makedirs(folder, exist_ok=True)
                if 'en_men' in task:
                    pass
                elif '999' in task:
                    pass
                elif '353' in task:
                    pass
                elif 'fern' in task:
                    ymin = -.05
                    ymax = .2
                elif 'dirani' in task:
                    ymin = -.02
                    ymax = .15
                elif 'kaneshiro' in task:
                    ymin = -.02
                    ymax = .1
                elif 'abstract' in task:
                    pass
                elif 'anew-lexical' in task:
                    ymin = -0.05
                    ymax = 0.4
                elif 'anew-word' in task:
                    ymin = -0.05
                    ymax = 0.2
                elif 'lexical' in task:
                    ymin = -0.05
                    ymax = 0.35
                elif 'naming' in task:
                    ymin = -0.05
                    ymax = 0.15
                elif 'abs-conc-decision' in task:
                    ymin = -.1
                    ymax = .1
                elif 'social-quant' in task:
                    ymin = -.25
                    ymax = .25
                elif 'mitchell' in task:
                    ymin = -.05
                    ymax = .15
                elif 'sem-phon' in task:
                    ymin = -0.05
                    ymax = 0.5
                elif 'sound-act' in task:
                    ymin = -.3
                    ymax = .3
                elif 'distr-learn' in task:
                    ymin = -.1
                    ymax = .3
                elif 'pmtg-prod' in task:
                    ymin = -0.02
                    ymax = 0.36
                fig, ax = pyplot.subplots(
                                          constrained_layout=True,
                                          figsize=(20, 10),
                                          )
                ### dividing into lines and regular values
                others = {
                          k : sorted(vals.items(), key=lambda item : item[0]) \
                                  for k, vals in t_res.items()
                                  }
                ### plotting horizontal lines
                all_vals = [val[0] for v in others.values() for val in v] + [0.]
                ax.hlines(
                          y=0,
                          xmin=-.1,
                          xmax=max(all_vals)+.1,
                          color='black',
                          )
                ### llms
                ### we use rainbow as a set of colours to sample from
                #print('count')
                colors = {k : v for k, v in zip(others.keys(), matplotlib.cm.rainbow(numpy.linspace(0, 1, len(others.keys()))))}
                for case, sort_freqs in others.items():
                    xs_freqs = [v[0] for v in sort_freqs]
                    ys_freqs = [v[1] for v in sort_freqs]
                    ys = [numpy.average(v) for v in ys_freqs]
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
                ### setting plot limits
                ax.set_ylim(
                            #bottom=min(0, min(alls))-corr, 
                            #top=max(0, max(alls))+corr
                            bottom=ymin,
                            top=ymax,
                            )
                #ax.set_xlim(
                            #left=0., 
                            #right=500000,
                            #right
                #            )
                #pyplot.ylabel('Pearson correlation')
                pyplot.ylabel('Spearman correlation')
                ### legend
                pyplot.legend(
                        ncols=5,
                        loc=8,
                        fontsize=10,
                        )
                ### saving figure
                #print('saving')
                pyplot.savefig(
                                os.path.join(
                                             folder,
                                     '{}.jpg'.format(task))
                                )
                #print('saved')
                pyplot.clf()
                pyplot.close()
                print(os.path.join(folder, task))
                counter.update(1)
