import argparse
import matplotlib
import numpy
import os
import random
import scipy
import sys

from matplotlib import pyplot
from utf_utils import transform_german_word


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
