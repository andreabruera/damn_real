import os
import sys

sys.path.append('..')
sys.path.append('.')

from test_utils import args, load_dataset
from psycholing_norms_loaders import load_lancaster_en_de_it

args = args()
if 'fern' in args.dataset:
    lancaster_ratings, trans_from_en = load_lancaster_en_de_it(args)
else:
    trans_from_en = {}

words, datasets = load_dataset(args, trans_from_en)

out_f = os.path.join('collect_word_sentences', 'words', args.lang, )
os.makedirs(out_f, exist_ok=True)
with open(os.path.join(out_f, '{}#words.tsv'.format(args.dataset)), 'w') as o:
    for w in words:
        o.write('{}\n'.format(w))
