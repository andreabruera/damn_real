import matplotlib
import mne
import numpy
import os
import random
import re
import scipy

from matplotlib import colormaps, font_manager, pyplot
from mne import stats
from scipy import stats

from plot_utils import font_setup
from tqdm import tqdm

def permutation_two_samples(one, two):
    one = one.tolist()
    assert len(one) == 1000
    two = two.tolist()
    assert len(two) == 1000
    ### permutation test
    real_diff = abs(numpy.average(one)-numpy.average(two))
    fake_distr = list()
    for _ in tqdm(range(1000)):
        fake = random.sample(one+two, k=len(one+two))
        fake_one = fake[:int(len(fake)*.5)]
        fake_two = fake[int(len(fake)*.5):]
        fake_diff = abs(numpy.average(fake_one)-numpy.average(fake_two))
        fake_distr.append(fake_diff)
    ### p-value
    p_val = (sum([1 for _ in fake_distr if _>real_diff])+1)/(1001)
    ### t-value
    # adapted from https://matthew-brett.github.io/cfd2020/permutation/permutation_and_t_test.html
    pers_errors = [v-numpy.average(one) for v in one]
    place_errors = [v-numpy.average(two) for v in two]
    all_errors = pers_errors + place_errors
    est_error_sd = numpy.sqrt(sum([er**2 for er in all_errors]) / (len(one) + len(two) - 2))
    sampling_sd_estimate = est_error_sd * numpy.sqrt(1 / len(one) + 1 / len(two))
    t_val = real_diff/sampling_sd_estimate

    return t_val, p_val

font_folder = '../../fonts'
font_setup(font_folder)

results = dict()

for root, direc, fz in os.walk(
                          os.path.join(
                              'test_results',
                              )):
    if 'fasttext' in root and 'alig' not in root:
        model = 'fasttext'
        pass
    elif 'gpt2_surprisal' in root:
        model = 'GPT2\nsurprisal'
        pass
    elif 'wac' in root and '200000' in root:
        model = 'Wac\nPPMI'
        pass
    elif 'wac' in root and 'abs-prob' in root:
        model = 'Wac\nfrequency'
        pass
    elif 'wac' in root and 'surprisal' in root:
        model = 'Wac\nsurprisal'
        pass
    elif 'llama-3b' in root and 'best' in root:
        model = 'Llama-3.2 3b'
        pass
    else:
        continue
    for f in fz:
        if 'resid' not in f:
            continue
        with open(os.path.join(root, f)) as i:
            for l in i:
                line = l.strip().split('\t')
                lang = line[0]
                if lang not in results.keys():
                    results[lang] = dict()
                old_model = line[1]
                #if 'wac' in old_model and '200000' in old_model:
                #    num = int(model.split('_')[-2])
                #    short_model = '_'.join(model.split('_')[1:-2]+[str(num)])
                all_task = line[2]
                ### modality
                assert all_task[:3] == '{}_'.format(lang)
                task = all_task[3:].split('#')[0].split('_')[0]
                if 'sem' in task or 'pmtg' in task:
                    splitter = '-'
                else:
                    splitter = '_'
                case = all_task.split('#')[-1].split(splitter)[0]
                cond = all_task.split('#')[-1].split(splitter)[-1]
                cond = '{}{}'.format(cond[0].lower(), cond[1:])
                if cond == 'cedx':
                    cond = 'rCereb'
                if cond == 'cz':
                    cond = 'vertex'
                if cond not in ['sham', 'vertex']:
                    cond = 'TMS\n{}'.format(cond)
                if 'distr-learn' in all_task:
                    pass
                elif 'pmtg-prod' in all_task:
                    pass
                elif 'sem-phon' in all_task:
                    pass
                elif 'sound-act' in all_task:
                    if 'all-pos' not in all_task:
                        continue
                    if 'all_all' in all_task:
                        continue
                    if 'detailed' in all_task:
                        continue
                    case = '{}-{}'.format(case, all_task.split('_')[-2])
                    pass
                elif 'social' in all_task:
                    if 'prime-cat' not in all_task:
                        continue
                    if 'cong' in all_task:
                        continue
                    pass
                else:
                    continue
                #print(all_task)
                if task not in results[lang].keys():
                    results[lang][task] = dict()
                if case not in results[lang][task].keys():
                    results[lang][task][case] = dict()
                if cond not in results[lang][task][case].keys():
                    results[lang][task][case][cond] = dict()
                non_nan_res = [v if v!='nan' else 0. for v in line[3:]]
                res = numpy.array(non_nan_res, dtype=numpy.float32)
                #if 'wac' in model and '200000' in model:
                #    results[lang][task][case][cond][short_model] = res
                #else:
                results[lang][task][case][cond][model] = res

colors = [
          ('navy','lightsteelblue',  'royalblue',),
          ('seagreen', 'mediumaquamarine', 'mediumseagreen'),
          ('lightskyblue', 'lightblue', 'paleturquoise'),
          ('mediumvioletred', 'pink', 'palevioletred'),
          ('mediumorchid', 'thistle', 'plum'),
          ('darkkhaki', 'wheat', 'khaki'),
          ]

out_f = 'paper_bars'
os.makedirs(out_f, exist_ok=True)

for lang, l_results in results.items():
    for task, t_results in l_results.items():
        for case, c_results in t_results.items():
            curr_fold = os.path.join(out_f, lang, task, case)
            os.makedirs(curr_fold, exist_ok=True)
            conds = sorted(c_results.keys(), reverse=True)
            models = set([m for _ in c_results.values() for m in _.keys()])
            sorted_models = ['Wac\nPPMI', 'fasttext', 'Llama-3.2 3b'] +\
                            sorted([m for m in models if 'surp' in m], reverse=True) +\
                            [m for m in models if 'freq' in m]
                            #sorted([m for m in models if 'surpr' not in m and 'freq' not in m], reverse=True) +\
            print(sorted_models)
            assert len(sorted_models) == len(models)
            xs = list(range(len(models)))
            if len(conds) == 2:
                corrections = list(numpy.linspace(-.33, .33, len(conds)))
                txt_corrections = list(numpy.linspace(-.4, .4, len(conds)))
                m_sc = 2000
                t_s = 20
            else:
                corrections = list(numpy.linspace(-.5, .5, len(conds)))
                txt_corrections = list(numpy.linspace(-.55, .55, len(conds)))
                m_sc = 1400
                t_s = 15
            fig, ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10))
            x_shift = 0
            xticks = list()
            counter = -1
            ps = list()
            for m_i, m in enumerate(sorted_models):
                if 'Resp' in m:
                    gen_avg = numpy.average([v for _ in c_results.values() for __ in _.values() for v in __])
                    gen_std = numpy.std([v for _ in c_results.values() for __ in _.values() for v in __])
                counter += 1
                for c_i, c in enumerate(conds):
                    color=colors[m_i][c_i]
                    xticks.append((counter, m))
                    
                    if len(conds) == 2:
                        w = 0.6
                    else:
                        w = 0.45
                    if c_i == 0:
                        for other_i, other in enumerate(conds[1:]):
                            two = c_results[other][m]
                            t_val, p_val = permutation_two_samples(c_results[c][m], two)
                            ps.append((m, (c, other), p_val, t_val))
                    ### simple p-value
                    p = (sum([1 for _ in c_results[c][m] if _<0.])+1)/(len(c_results[c][m])+1)
                    #print(p)
                    ps.append((m, c, p))
                    ### bar
                    ax.bar(
                           m_i+corrections[c_i]+x_shift, 
                           numpy.average(c_results[c][m]),
                           width=w,
                           color=color,
                           edgecolor='gray',
                           zorder=2.
                           )
                    ax.scatter(
                           [m_i+corrections[c_i]+x_shift+(random.randrange(-m_sc, m_sc)*0.0001) for rand in range(len(c_results[c][m]))], 
                           c_results[c][m],
                           color=color,
                           edgecolor='white',
                           alpha=0.2,
                           zorder=2.5
                           )
                    ax.text(
                           m_i+txt_corrections[c_i]+x_shift, 
                           -.08,
                           s=c,
                           fontsize=t_s,
                           ha='center',
                           va='center',
                           )
                x_shift += 1
                counter += 1
                if m_i in [2, 4]:
                    x_shift += 1
                    counter += 1
            ### absolute p-values
            corr_ps = mne.stats.fdr_correction([v[2] for v in ps])[1]
            assert len(corr_ps) == len(ps)
            corr_ps = [(ps[i][0], ps[i][1], p) for i, p in enumerate(corr_ps) if type(ps[i][1]!=tuple)]
            x_shift = 0
            counter = -1
            for m_i, m in enumerate(sorted_models):
                counter += 1
                for c_i, c in enumerate(conds):
                    for pm, pc, pp in corr_ps:
                        if pm == m and pc == c:
                            if pp < 0.0005:
                                print(pp)
                                ax.scatter(
                                       m_i+corrections[c_i]+x_shift, 
                                       0.015,
                                       color='black',
                                       edgecolor='white',
                                       zorder=3.,
                                       marker='*',
                                       s=300
                                       )
                                ax.scatter(
                                       m_i+corrections[c_i]+x_shift+.1, 
                                       0.01,
                                       color='black',
                                       edgecolor='white',
                                       zorder=3.,
                                       marker='*',
                                       s=300
                                       )
                                ax.scatter(
                                       m_i+corrections[c_i]+x_shift-.1, 
                                       0.01,
                                       color='black',
                                       edgecolor='white',
                                       zorder=3.,
                                       marker='*',
                                       s=300
                                       )
                            elif pp < 0.005:
                                print(pp)
                                ax.scatter(
                                       m_i+corrections[c_i]+x_shift-.075, 
                                       0.01,
                                       color='black',
                                       edgecolor='white',
                                       zorder=3.,
                                       marker='*',
                                       s=300
                                       )
                                ax.scatter(
                                       m_i+corrections[c_i]+x_shift+.075, 
                                       0.01,
                                       color='black',
                                       edgecolor='white',
                                       zorder=3.,
                                       marker='*',
                                       s=300
                                       )
                            elif pp < 0.05:
                                print(pp)
                                ax.scatter(
                                       m_i+corrections[c_i]+x_shift, 
                                       0.01,
                                       color='black',
                                       edgecolor='white',
                                       zorder=3.,
                                       marker='*',
                                       s=300
                                       )
                            elif pp < 0.1:
                                print(pp)
                                ax.scatter(
                                       m_i+corrections[c_i]+x_shift, 
                                       0.01,
                                       color='black',
                                       edgecolor='white',
                                       zorder=3.,
                                       marker='v',
                                       s=100
                                       )
                x_shift += 1
                counter += 1
                if m_i in [2, 4]:
                    x_shift += 1
                    counter += 1
            ### relative p-values
            corr_ps = mne.stats.fdr_correction([v[2] for v in ps])[1]
            corr_ps = [(ps[i][0], ps[i][1], p, ps[i][3]) for i, p in enumerate(corr_ps) if type(ps[i][1])==tuple]
            x_shift = 0
            counter = -1
            for m_i, m in enumerate(sorted_models):
                counter += 1
                for c_i, c in enumerate(conds):
                    for pm, pc, pp, t in corr_ps:
                        if pm == m and pc[0] == c:
                            for other_i, other in enumerate(conds[1:]):
                                if pc[1] == other:
                                    color=colors[m_i][other_i+1]
                                    if pp < 0.1:
                                        ax.vlines(
                                                  ymin=0.31-(other_i*0.03), 
                                                  ymax=0.31-(other_i*0.03)-0.01, 
                                                  x=m_i+corrections[c_i]+x_shift,
                                                  color=color,
                                                  linewidth=5.
                                                  )
                                        ax.vlines(
                                                  ymin=0.31-(other_i*0.03), 
                                                  ymax=0.31-(other_i*0.03)-0.01, 
                                                  x=m_i+corrections[c_i+1+other_i]+x_shift,
                                                  color=color,
                                                  linewidth=5.
                                                  )
                                        ax.hlines(
                                                  xmin=m_i+corrections[c_i]+x_shift-0.025, 
                                                  xmax=m_i+corrections[c_i+1+other_i]+x_shift+0.03, 
                                                  y=0.31-(other_i*0.03),
                                                  color=color,
                                                  linewidth=5.
                                                  )
                                    xmin=m_i+corrections[c_i]+x_shift 
                                    xmax=m_i+corrections[c_i+1+other_i]+x_shift 
                                    middle = xmin + ((xmax-xmin)*.5)
                                    if pp < 0.0005:
                                        print(pp)
                                        ax.scatter(
                                                   middle,
                                                   y=0.32-(other_i*0.03),
                                                      color=color,
                                                   edgecolor='gray',
                                                   zorder=3.,
                                                   marker='*',
                                                   s=300
                                                   )
                                        ax.scatter(
                                                   middle+.1,
                                                   y=0.32-(other_i*0.03),
                                                      color=color,
                                                   edgecolor='gray',
                                                   zorder=3.,
                                                   marker='*',
                                                   s=300
                                                   )
                                        ax.scatter(
                                                   middle-.1,
                                                   y=0.32-(other_i*0.03),
                                                      color=color,
                                                   edgecolor='gray',
                                                   zorder=3.,
                                                   marker='*',
                                                   s=300
                                                   )
                                    elif pp < 0.005:
                                        print(pp)
                                        ax.scatter(
                                                   middle-0.075,
                                                   y=0.32-(other_i*0.03),
                                                   color=color,
                                                   edgecolor='gray',
                                                   zorder=3.,
                                                   marker='*',
                                                   s=300
                                                   )
                                        ax.scatter(
                                                   middle+0.075,
                                                   y=0.32-(other_i*0.03),
                                                   color=color,
                                                   edgecolor='gray',
                                                   zorder=3.,
                                                   marker='*',
                                                   s=300
                                                   )
                                    elif pp < 0.05:
                                        ax.scatter(
                                                   middle,
                                                   y=0.32-(other_i*0.03),
                                                   color=color,
                                                   edgecolor='gray',
                                                   zorder=3.,
                                                   marker='*',
                                                   s=300
                                                   )
                                    elif pp < 0.1:
                                        ax.scatter(
                                                   middle,
                                                   y=0.32-(other_i*0.03),
                                                   color=color,
                                                   edgecolor='gray',
                                                   zorder=3.,
                                                   marker='v',
                                                   s=100
                                                   )
                x_shift += 1
                counter += 1
                if m_i in [2, 4]:
                    x_shift += 1
                    counter += 1
            ax.set_ylim(bottom=-.095, top=.36)
            ax.text(
                    2.,
                    0.34,
                    s='Word vector similarity',
                    fontsize=25,
                    fontweight='bold',
                    ha='center',
                    )
            ax.text(
                    8.,
                    0.34,
                    s='Surprisal',
                    fontsize=25,
                    fontweight='bold',
                    ha='center',
                    )
            ax.text(
                    12.,
                    0.34,
                    s='Word frequency',
                    fontsize=25,
                    fontweight='bold',
                    ha='center',
                    )

            ax.hlines(xmin=-1, xmax=len(models)+x_shift-1, color='black', y=0)
            ax.hlines(xmin=-1, xmax=len(models)+x_shift-1, color='silver',alpha=0.5,linestyle='dashed', y=[y*0.01 for y in range(-5, 35, 5)], zorder=1)
            pyplot.ylabel('Spearman correlation (RSA RT-model)', fontsize=23)
            pyplot.xticks(
                          [x[0] for x in xticks], 
                          [x[1] for x in xticks],
                          fontsize=25,
                          fontweight='bold')
            pyplot.savefig(os.path.join(curr_fold, '{}.jpg'.format(case)), dpi=300)
            pyplot.savefig(os.path.join(curr_fold, '{}.svg'.format(case)),)
