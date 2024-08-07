import matplotlib
import numpy
import os
import re
import scipy

from matplotlib import pyplot
from tqdm import tqdm

results = dict()
for dataset in [1, 2]:
    results_folder = os.path.join(
                                  'results', 
                                  'fernandino{}'.format(dataset),
                                  'full_dataset',
                                  )
    for mode in os.listdir(results_folder):
        if mode not in results.keys():
            results[mode] = dict()
        if dataset not in results[mode].keys():
            results[mode][dataset] = dict()
        for damage in os.listdir(os.path.join(results_folder, mode)):
            full_path = os.path.join(results_folder, mode, damage)
            fs = os.listdir(full_path)
            if damage == 'undamaged':
                fs = [f for f in fs if 'pmi75_opensubs' in f and 'de_' not in f]
            else:
                if 'domin' not in damage:
                    continue
            assert len(fs) == 1
            with open(os.path.join(full_path, fs[0])) as i:
                for l in i:
                    line = l.strip().split('\t')
                    #print(line)
                    area = line[0].replace('_seman', 'seman')
                    if area not in results[mode][dataset].keys():
                        results[mode][dataset][area] = dict()
                    #model = f.replace('.txt', '')
                    scores = numpy.array(line[1:], dtype=numpy.float64)
                    #results[dataset].append([(area, model), scores])
                    results[mode][dataset][area][damage] = scores
cmap = matplotlib.colormaps['hsv']
### plotting

with tqdm() as counter:
    for mode, mode_data in results.items():
        for dataset, dataset_data in mode_data.items():
            plots = os.path.join(
                                      'plots', 
                                      'fernandino{}'.format(dataset),
                                      'full_dataset',
                                      mode,
                                      'damaged',
                                      )
            os.makedirs(plots, exist_ok=True)
            for area, area_data in dataset_data.items():
                colors = cmap(numpy.linspace(0.,1,len(area_data.keys())))
                if len(area_data.keys()) == 1:
                    continue
                fig, ax = pyplot.subplots(constrained_layout=True)
                ymin=-7
                if 'individual' in mode:
                    ymax = 10
                else:
                    ymax = 11
                hlines=[y*0.01 for y in range(ymin+1, ymax, 2)]
                ax.set_ylim(bottom=ymin*0.01, top=ymax*0.01)
                models = sorted(area_data.keys())
                #print(models)
                ax.hlines(
                          y=[0.0],
                          xmin=-.2,
                          xmax=len(models)-.8,
                          alpha=0.3,
                          linestyles='-',
                          color='grey',
                          )
                ax.hlines(
                          #y=[-.06, -0.04, -0.02, 0.02, 0.04, 0.06, 0.08, .1, .12, .14, .16, .18, .2,],
                          y=hlines,
                          xmin=-.2,
                          xmax=len(models)-.8,
                          alpha=0.1,
                          linestyles='--',
                          )
                plot_colors = {k : colors[k_i] for k_i, k in enumerate(models)}
                for m_i, m in enumerate(models):
                    if 'individual' in mode:
                        vi = ax.violinplot(
                                      area_data[m],
                                      [m_i],
                                      showextrema=False,
                                      )
                        for b in vi['bodies']:
                            vi_m = numpy.mean(b.get_paths()[0].vertices[:, 0])
                            b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], -numpy.inf, vi_m)
                            b.set_color(plot_colors[m])
                    else:
                        assert len(area_data[m]) == 1
                        ax.bar(
                               m_i,
                               area_data[m][0],
                               color=plot_colors[m],
                               edgecolor='silver'
                               )
                    ax.scatter(
                                           m_i,
                                           numpy.average(area_data[m]),
                                           color=plot_colors[m],
                                           edgecolors='white',
                                           marker='D',
                                           )
                x_models = [re.sub(r'dominance|pow_0\.', '', m) for m in models]
                pyplot.xticks(
                          ticks=range(len(models)), 
                          labels=x_models,
                          #rotation=45,
                          rotation=90,
                          )
                pyplot.title(area)
                pyplot.savefig(os.path.join(plots, '{}.jpg'.format(area)))
                pyplot.clf()
                pyplot.close()
                counter.update(1)
