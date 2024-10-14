import os
import sys

sys.path.append('..')
sys.path.append('.')

from utf_utils import transform_german_word
from test_utils import args, load_dataset
from psycholing_norms_loaders import load_lancaster_en_de_it

from tqdm import tqdm

args = args()
if 'fern' in args.dataset:
    lancaster_ratings, trans_from_en = load_lancaster_en_de_it(args)
else:
    trans_from_en = {}

words, datasets = load_dataset(args, trans_from_en)
trials = list(set([k[0] for _ in datasets.values() for __ in _.values() for k in __]))
if type(trials[0]) == str:
    trials = [(t_one, t_two) for t_one_i, t_one in enumerate(trials) for t_two_i, t_two in enumerate(trials) if t_two_i!=t_one_i]

out_f = os.path.join('collect_word_sentences', 'trials', args.lang, )
os.makedirs(out_f, exist_ok=True)
with tqdm() as counter:
    with open(os.path.join(out_f, '{}#trials.tsv'.format(args.dataset)), 'w') as o:
        for trial in trials:
            if '_' in trial[0]:
                t = (trial[0].split('_')[-1], trial[1].split('_')[-1])
            else:
                t = (w for w in trial)
            print(t)
            ### word one
            if type(t[0]) == str:
                if args.lang in ['it', 'en']:
                    w_one = [t[0].lower(),]
                else:
                    if '_' in t[0]:
                        c_w = t[0].split('_')[-1]
                    else:
                        c_w = t[0]
                    w_one = transform_german_word(c_w, lowercase=False)
            else:
                if args.lang in ['it', 'en']:
                    w_one = set([ind for w in t[0] for ind in [w.lower(), w.capitalize()]])
                else:
                    if '_' in t[0][0]:
                        c_w = [w.split('_')[-1] for w in t[0]]
                    else:
                        c_w = t[0]
                    w_one = set([ind for w in t[0] for ind in transform_german_word(c_w, lowercase=False)])
            ### word two
            if type(t[1]) == str:
                if args.lang in ['it', 'en']:
                    w_two = [t[1].lower()]
                else:
                    if '_' in t[1]:
                        c_w = t[1].split('_')[-1]
                    else:
                        c_w = t[1]
                    w_two = transform_german_word(t[1], lowercase=False)
            else:
                if args.lang in ['it', 'en']:
                    w_two = set([ind for w in t[1] for ind in [w.lower(), w.capitalize()]])
                else:
                    if '_' in t[1][0]:
                        c_w = [w.split('_')[-1] for w in t[1]]
                    else:
                        c_w = t[1][0]
                    w_two = set([ind for w in t[1] for ind in transform_german_word(w, lowercase=False)])
            for on in w_one:
                if on[0].isupper() == False:
                    continue
                for tw in w_two:
                    if tw[0].isupper() == False:
                        continue
                    o.write('{}\t{}\n'.format(on, tw))
                    counter.update(1)