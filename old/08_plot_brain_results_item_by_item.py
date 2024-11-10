import matplotlib
import numpy
import os
import pdb
import scipy

from matplotlib import pyplot
from scipy import stats

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

damages = [
           #'full_random',
           #'noise_injection',
           #'auditory_relu-raw-thresholded9095_random',
           #'auditory_relu-raw90_random',
           #'auditory_relu-raw75_random',
           'auditory_relu-raw95_random',
           #'auditory_relu-exponential75_random',
           #'auditory_relu-exponential90_random',
           #'auditory_relu-exponential95_random',
           ]

plot_folder = os.path.join('brain_plots', 'rsa', 'item_by_item')
os.makedirs(plot_folder, exist_ok=True)

for damage in damages:
    print('\n')
    for dataset in ['1', '2']:
        ### open results
        dam_words = list()
        damaged = list()
        #with open(os.path.join('brain_results', 'rsa_fernandino1_w2v_en_auditory_relu-raw95_random.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino1_w2v_en_auditory_relu-raw-thresholded9095_random.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino2_w2v_en_auditory_relu-raw-thresholded9095_random.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino2_w2v_en_auditory_relu-raw-thresholded8595_random.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino2_w2v_en_noise_injection.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino1_w2v_en_noise_injection.results')) as i:
        with open(os.path.join('brain_results', 'rsa_fernandino{}_w2v_en_{}.results'.format(dataset, damage))) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                dam_words.append(line[0])
                damaged.append(line[1:])
        damaged = numpy.array(damaged, dtype=numpy.float64)
        ### open results
        words = list()
        undamaged = list()
        #with open(os.path.join('brain_results', 'fernandino2_undamaged_w2v_en.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino1_undamaged_w2v_en.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino2_undamaged_w2v_en.results')) as i:
        with open(os.path.join('brain_results', 'rsa_fernandino{}_undamaged_w2v_en.results'.format(dataset))) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                if line[0] in dam_words:
                    words.append(line[0])
                    undamaged.append(line[1:])
        undamaged = numpy.array(undamaged, dtype=numpy.float64)

        assert dam_words == words
        diff = scipy.stats.ttest_rel(
                                    numpy.average(damaged, axis=0), 
                                    numpy.average(undamaged, axis=0), 
                                    alternative='less'
                                    #alternative='greater'
                                    )
        #diff = scipy.stats.wilcoxon(damaged[high_auditory, :].flatten(), undamaged[high_auditory, :].flatten(), alternative='less')
        print(damage)
        print(dataset)
        print(diff)
        ### plotting
        fig, ax = pyplot.subplots(constrained_layout=True)
        plot_modalities = [
                 'Auditory.mean',
                 #'Gustatory.mean',
                 #'Haptic.mean',
                 #'Olfactory.mean',
                 #'Visual.mean',
                 'Mouth.mean', 
                 ]
        plot_colors = {
                 'Auditory.mean' : 'teal',
                 'Gustatory.mean' : 'orange',
                 'Haptic.mean' : 'goldenrod',
                 'Olfactory.mean' : 'grey',
                 'Visual.mean' : 'magenta',
                 'Mouth.mean' : 'coral', 
                 }
        for mod in plot_modalities:
            plot_idx = relevant_keys.index(mod)
            relevant_norms = [norms[k][plot_idx] for k in dam_words]
            sorted_idxs = [k[0] for k in sorted(enumerate(relevant_norms), key=lambda item : item[1])]
            mod_xs = [relevant_norms[i] for i in sorted_idxs]
            mod_dam_ys = [numpy.average(damaged[i, :]) for i in sorted_idxs]
            mod_undam_ys = [numpy.average(undamaged[i, :]) for i in sorted_idxs]
            ax.plot(mod_xs, mod_dam_ys, color=plot_colors[mod], label=mod.split('.')[0])
            ax.plot(mod_xs, mod_undam_ys, color=plot_colors[mod], ls='-.')
        ax.set_xlim(left=0., right=5.)
        ax.legend()
        file_out = os.path.join(plot_folder, 'brain_rsa_fernandino{}_{}.jpg'.format(dataset, damage))
        print(file_out)
        pyplot.savefig(file_out)
        pyplot.clf()
        pyplot.close()
