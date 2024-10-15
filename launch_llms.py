import argparse
import os

#stat_approach = 'simple'
#stat_approach = 'bootstrap'
stat_approach = 'residualize'

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
                #'de_abstract-fmri',
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
            #'ws353',
            'en_men',
            ]
elif args.modality == 'behav':
    datasets = [
                ### behav
                'de_behav',
                #'it_behav',
                #'it_mouse',
                'it_deafhearing',
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
    
    choices = list()
    llms = [
         #'gpt2',
         #'llama-1b',
         'llama-3b',
         #'xglm-564m',
         #'xglm-1.7b',
         #'xglm-2.9b',
         #'xglm-4.5b',
         #'xglm-7.5b',
         #'xlm-roberta-large',
         #'xlm-roberta-xl',
         #'xlm-roberta-xxl',
         ]
    for llm in llms:
        if '1b' in llm:
            m = 16
        elif '3b' in llm:
            m = 28
        elif 'erta-xl' in llm:
            m = 36
        elif '2.9' in llm:
            m = 48
        elif '4.5' in llm:
            m = 48
        elif '7.5' in llm:
            m = 48
        elif 'erpt' in llm:
            m = 36
        else:
            m = 24
        #for l in range(m):
        #    choices.append('{}_layer-{}'.format(llm, l))
        #choices.append('{}_surprisal'.format(llm))
        choices.append('{}_best'.format(llm))
    for model in choices:
        os.system('python3 test.py --lang {} --model {} --dataset {} --stat_approach {}'.format(args.lang, model, dataset, stat_approach))
