import matplotlib
import nilearn
import numpy
import os

from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from nilearn import datasets, plotting, surface

fsaverage = datasets.fetch_surf_fsaverage()
dataset = nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
#dataset =nilearn.datasets.fetch_atlas_aal(version='SPM12')
maps = dataset['maps']
maps_data = nilearn.image.load_img(maps).get_fdata()
idxs = [float(v_i) for v_i in range(len(dataset['labels']))]
labels = dataset['labels']
assert len(idxs) == len(labels)
rel_names = {
            'aifg' : [
                'Inferior Frontal Gyrus, pars triangularis'
                    ],
            'pifg' : [
                'Inferior Frontal Gyrus, pars opercularis',
                      ],
            'pipl' : [
                'Supramarginal Gyrus, posterior division', 'Angular Gyrus'
                      ],
            'satl' : [
                'Middle Temporal Gyrus, anterior division',
                'Superior Temporal Gyrus, anterior division',
                'Temporal Pole'
                      ],
            'ips' : [
                'Superior Parietal Lobule'
                      ],
            'pmtg' : [
                'Middle Temporal Gyrus, posterior division', 'Middle Temporal Gyrus, temporooccipital part'
                      ],
            'cereb' : [
                      ]
             }
inverse = {v : k for k, _ in rel_names.items() for v in _}

colormaps = {
             'ifg' : 'BuGn',
             'ptl' : 'RdPu',
             'sma' : 'Blues',
             'dlpfc' : 'YlOrBr',
             }
vals = {
        'aifg' :  {'max' : 1., 'min' : 0.86},
        'pifg' : {'max' : 1., 'min' : 0.86},
        'ips' : {'max' : 1., 'min' : 0.86},
        'pipl' : {'max' : 1., 'min' : 0.86},
        'satl' : {'max' : 1., 'min' : 0.86},
        'pmtg' : {'max' : 1., 'min' : 0.86},
        'cereb' : {'max' : 1., 'min' : 0.86},
             }
### generic cmap
cmaps=dict()
colors=[
        'white',
        'mediumseagreen',
        'white',
        'goldenrod',
        'white',
        'mediumvioletred',
        'white',
        'dodgerblue',
        ]
cmap = LinearSegmentedColormap.from_list("mycmap", colors)
cmaps['symmetric'] = cmap
### right
right=[
       'white',
       'mediumvioletred',
       'white',
       'forestgreen',
       ]
right_cmap = LinearSegmentedColormap.from_list("right", right)
### left
left=[
      #'white',
      #'paleturquoise',
      #'white',
      #'lightpink',
      #'white',
      #'khaki',
      'white',
      #'forestgreen',
      #'mediumvioletred',
      'orange',
      #'steelblue',
      ]
left_cmap = LinearSegmentedColormap.from_list("left", left)
cmaps = [
         #pers_cmap,
         #'cividis',
         'YlOrBr',
         #'coolwarm',
         ]
#left_cmap = 'YlOrBr',
for area, labels_names in rel_names.items():
    relevant_labels = {float(labels.index(k)) : inverse[k] for k in labels_names}
    out = os.path.join(
                       'region_maps',
                       )
    os.makedirs(
                out,
                exist_ok=True,
                )
    for view in [
                 'lateral',
                 #'ventral',
                 #'dorsal',
                 #'anterior',
                 ]:
        alpha = 0.1
        #relevant_labels = {i : vals[k]['max'] for i, l in enumerate(labels) for k, v in relevant_labels.items() if l in v}
        msk = numpy.array([vals[relevant_labels[v]]['max'] if float(v) in relevant_labels.keys() else 0. for v in maps_data.flatten()]).reshape(maps_data.shape)
        atl_img = nilearn.image.new_img_like(maps, msk)
        cmap = left_cmap
        print(sum(msk.flatten()))
        ### Right
        texture = surface.vol_to_surf(
                                      atl_img,
                                      fsaverage.pial_right,
                                      interpolation='nearest',
                                      radius=0.,
                                      n_samples=1,
                                      )
        r= plotting.plot_surf_stat_map(
                fsaverage.pial_right,
                texture,
                hemi='right',
                title='{} - right hemisphere'.format(area),
                threshold=0.01,
                colorbar=True,
                bg_on_data=False,
                #bg_on_map=False,
                bg_map=None,
                darkness=0.6,
                alpha=alpha,
                view=view,
                cmap=cmap,
                vmin=0.,
                vmax=1.
                )
        #pyplot.savefig(os.path.join(out, \
        r.savefig(os.path.join(
               out,
               'surface_right_{}_{}.svg'.format(
                                           area,
                                           view,
                                           ),
                ),
                dpi=600
                )
        #pyplot.savefig(os.path.join(out, \
        r.savefig(os.path.join(
                out,
                'surface_right_{}_{}.jpg'.format(
                                           area,
                                           view,
                                           ),
                ),
                dpi=600
                )
        pyplot.clf()
        pyplot.close()
        ### Left
        texture = surface.vol_to_surf(
                                      atl_img,
                                      fsaverage.pial_left,
                                      interpolation='nearest',
                                      radius=0.,
                                      n_samples=1,
                )
        l = plotting.plot_surf_stat_map(
                fsaverage.pial_left,
                texture,
                hemi='left',
                title='{} - left hemisphere'.format(area),
                colorbar=True,
                threshold=0.01,
                bg_on_data=False,
                #bg_on_map=False,
                bg_map=None,
                cmap=cmap,
                view=view,
                darkness=0.5,
                alpha=alpha,
                vmin=0.,
                vmax=1.
                )
        l.savefig(os.path.join(out, \
                    'surface_left_{}_{}.svg'.format(area, view),
                    ),
                    dpi=600
                    )
        l.savefig(os.path.join(out, \
                    'surface_left_{}_{}.jpg'.format(area, view),
                    ),
                    dpi=600
                    )
        pyplot.clf()
        pyplot.close()
