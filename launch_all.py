import argparse
import os

stat_approach = 'simple'
#stat_approach = 'bootstrap'
#stat_approach = 'residualize'

parser = argparse.ArgumentParser()
parser.add_argument(
                     '--lang', 
                     choices=['de', 'it', 'en'],
                     required=True,
                     )
parser.add_argument(
                     '--modality', 
                     choices=['behav', 'tms', 'fmri', 'meeg', 'simrel'],
                     required=True,
                     )
args = parser.parse_args()

if args.modality == 'fmri':
    datasets = [
                ### fmri
                #'fern2-all',
                #'fern2-areas-all',
                #'fern1-categories',
                #'fern1-all',
                #'fern1-areas-all',
                #'fern2-categories',
                'de_abstract-fmri',
                'mitchell2008',
            ]
elif args.modality == 'tms':
    datasets = [
            ### tms
            'de_sem-phon',
            'de_pmtg-prod',
            'de_sound-act',
            'it_distr-learn',
            'it_social-quantity',
            ]
elif args.modality == 'meeg':
    datasets = [
            ### meeg
            'dirani-n400',
            'kaneshiro-n400'
            ]
elif args.modality == 'simrel':
    datasets = [
            ### sim-lex norms
            'simlex999',
            'ws353',
            'en_men',
            ]
elif args.modality == 'behav':
    datasets = [
                ### behav
                'de_behav',
                #'it_behav',
                #'it_mouse',
                #'it_deafhearing',
                'it_blindsighted',
                #'picture-naming-seven',
                'it_anew',
                ]
final_datasets = list()
for d in datasets:
    if d[2] == '_':
        if d[:3] == '{}_'.format(args.lang):
            final_datasets.append(d)
    else:
        final_datasets.append(d)

for dataset in final_datasets:
    
    corpora_choices = list()
    for corpus in [
                   'wac',
                   'opensubs',
                   'cc100',
                   'tagged_leipzig',
                   'tagged_wiki',
                   'tagged_gutenberg',
                   ]:
        #corpora_choices.append('{}-ppmi-vecs'.format(corpus))
        for mode in [
                     #'neg-raw-abs-prob',
                     'neg-log10-abs-prob',
                     #'neg-sym-raw-cond-prob',
                     #'neg-fwd-raw-cond-prob',
                     #'neg-sym-log10-cond-prob',
                     #'surprisal',
                     ]:
            corpora_choices.append('{}-{}'.format(corpus, mode))

    choices=[
             'fasttext',
             #'fasttext_aligned',
             #'conceptnet',
             ] + corpora_choices
    for model in choices:
        os.system('python3 test.py --lang {} --model {} --dataset {} --stat_approach {}'.format(args.lang, model, dataset, stat_approach))
