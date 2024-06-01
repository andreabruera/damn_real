import matplotlib
import numpy
import os
import pdb

from matplotlib import pyplot

file_path = os.path.join(
                         'data',
                         'Lancaster_sensorimotor_norms_for_39707_words.tsv',
                         )
assert os.path.exists(file_path)
relevant_keys = [
                 'Auditory.mean',
                 'Gustatory.mean',
                 'Haptic.mean',
                 'Olfactory.mean',
                 'Visual.mean',
                 'Foot_leg.mean',
                 'Hand_arm.mean', 
                 'Head.mean', 
                 'Mouth.mean', 
                 'Torso.mean'
                 ]

norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.strip().split('\t')
        if counter == 0:
            header = line.copy()
            counter += 1
            continue
        assert len(line) == len(header)
        #marker = False
        for k in relevant_keys:
            if float(line[header.index(k)]) > 10:
                line[header.index(k)] = '.{}'.format(line[header.index(k)])
            #try:
            assert float(line[header.index(k)]) < 10
            #except AssertionError:
            #    #logging.info(line[0])
            #    marker = True
        #if marker:
        #    continue
        if len(line[0].split()) == 1:
            norms[line[0].lower()] = list()
            for k in relevant_keys:
                val = float(line[header.index(k)])
                norms[line[0].lower()].append(val)

### reading datasets
datasets = {
            'mitchell' : list(), 
            'pereira' : list(),
            'fernandino1' : list(),
            'fernandino2' : list(),
            }
for d in datasets.keys():
    with open(os.path.join('data', '{}_words.txt'.format(d))) as i:
        delimiter = '\n' if d!='mitchell' else ', '
        words = [w.strip() for w in i.read().strip().split(delimiter)]
        datasets[d] = words

### plotting stats

### checking that all words are there
for d, v in datasets.items():
    for w in v:
        if w not in norms.keys():
            print(w)

stats_datasets = {k : [val for val in v if val in norms.keys()] for k, v in datasets.items()}

folder = os.path.join('brain_plots', 'stats')
os.makedirs(folder, exist_ok=True)

### plot datasets
for dataset, data in stats_datasets.items():
    fig, ax = pyplot.subplots(figsize=(21,10), constrained_layout=True)
    ys = [[norms[d][k_i] for d in data] for k_i, k in enumerate(relevant_keys)]
    ax.violinplot(ys)
    ax.set_xticks([x+1. for x in range(len(relevant_keys))])
    ax.set_xticklabels(relevant_keys, fontsize=23, rotation=45, ha='center')
    pyplot.savefig(os.path.join(folder, '{}.jpg'.format(dataset)))
    pyplot.clf()
    pyplot.close()
