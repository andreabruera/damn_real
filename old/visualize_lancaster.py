import matplotlib
import numpy
import os
import scipy

from matplotlib import pyplot

file_path = os.path.join(
                         'data',
                         'Lancaster_sensorimotor_norms_for_39707_words.tsv',
                         )
assert os.path.exists(file_path)

norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.strip().split('\t')
        if counter == 0:
            header = line.copy()[1:]
            counter += 1
            continue
        assert len(line[1:]) == len(header)
        norms[line[0]] = line[1:]

### sound features
sound_dominant_key = 'Dominant.perceptual'
sound_dominant_words = [w for w, vals in norms.items() if vals[header.index(sound_dominant_key)]=='Auditory']

### finding words with highest association with sound
sound_key = 'Auditory.mean'
sound_norms = [float(vals[header.index(sound_key)]) for n, vals in norms.items()]
sound_norms = [v for v in sound_norms if v <= 5.]
removed_words = [w for w, vals in norms.items() if float(vals[header.index(sound_key)])>5.]
perc = numpy.percentile(sound_norms, 90)
sound_words = [w for w, vals in norms.items() if float(vals[header.index(sound_key)])>perc and w not in removed_words]

import pdb; pdb.set_trace()
