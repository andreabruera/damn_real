import matplotlib
import numpy
import os
import pdb
import scipy

from matplotlib import pyplot
from scipy import stats
from tqdm import tqdm

unit = 25
assert 100 % unit == 0
splits = [(i*0.01, (i+unit)*0.01) for i in range(0, 100, unit)]

with tqdm() as counter:
    for dataset in [1, 2]:
        results_folder = os.path.join(
                                      'results', 
                                      'fernandino{}'.format(dataset), 
                                      'bins',
                                      )
        assert os.path.exists(results_folder)
        results = dict()
        for root, direc, fz in os.walk(results_folder):
            for f in fz:
                if 'undamaged' not in f: 
                    continue
                split_root = root.split('/')
                marker= split_root[-3]
                model = '{}_{}'.format(marker, split_root[-4])
                area = split_root[-5]
                damage_variable = 'undamaged'
                mode = split_root[-7]
                if mode not in results.keys():
                    results[mode] = dict()
                if area not in results[mode].keys():
                    results[mode][area] = dict()
                with open(os.path.join(root, f)) as i:
                    for l_i, l in enumerate(i):
                        if l_i == 0:
                            continue
                        line = l.strip().split('\t')
                        curr_var =  line[0].split('_')[0]
                        case = f.replace('.results', '')
                        if 'nan' in line[1:]:
                            res = numpy.array([numpy.nan for _ in line[1:]])
                        else:
                            res = numpy.array(line[1:], dtype=numpy.float32)
                        if curr_var not in results[mode][area].keys():
                            results[mode][area][curr_var] = dict()
                        if model not in results[mode][area][curr_var].keys():
                            results[mode][area][curr_var][model] = list()
                        results[mode][area][curr_var][model].append(res)

        for mode, mode_d in results.items():
            if 'average'  in mode:
                ymin=-15
                ymax=27
            else:
                ymin=-9
                ymax=15
            hlines=[y*0.01 for y in range(ymin+1, ymax, 2)]
            for area, area_d in mode_d.items():
                for variable, variable_d in area_d.items():
                    plots_folder = os.path.join(
                                                'plots', 
                                                'fernandino{}'.format(dataset), 
                                                'bins',
                                                'undamaged',
                                                mode,
                                                area, 
                                                variable,
                                                )
                    os.makedirs(plots_folder, exist_ok=True)
                    ### plotting
                    fig, ax = pyplot.subplots(constrained_layout=True)
                    cmap = matplotlib.colormaps['Set3']
                    colors=cmap(numpy.linspace(0.,1,12))
                    ax.set_prop_cycle(color=colors)
                    xs = list(range(len(splits)))
                    for curr_model, c_model_d in variable_d.items():
                        #print(dam_results)
                        mod_ys = numpy.nanmean(c_model_d, axis=1)
                        ax.plot(xs, mod_ys, label=curr_model.split('_')[0])
                        ax.scatter(xs, mod_ys, marker='D', edgecolors='silver')
                    ax.set_xlim(left=-.5, right=len(splits)-.5)
                    ax.set_ylim(bottom=ymin*0.01, top=ymax*0.01)
                    ax.set_xticks(ticks=xs, labels=
                            ['{}-{}'.format(s[0], s[1]) for s in splits])
                    ax.hlines(
                              y=[0.0],
                              xmin=-.2,
                              xmax=len(splits)-.8,
                              alpha=0.3,
                              linestyles='-',
                              color='grey',
                              )
                    ax.hlines(
                              y=hlines,
                              xmin=-.2,
                              xmax=len(splits)-.8,
                              alpha=0.1,
                              linestyles='--',
                              )
                    ax.legend(ncols=3,
                              loc=4)
                    file_out = os.path.join(
                                    plots_folder, 
                                    '{}.jpeg'.format(variable),
                                    )
                    title = 'undamaged performance in: {}'.format(
                            area,
                            )
                    pyplot.title(label=title)
                    print(file_out)
                    pyplot.savefig(file_out)
                    pyplot.clf()
                    pyplot.close()
                    counter.update(1)
