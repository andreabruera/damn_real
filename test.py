from tqdm import tqdm

from fmri_loaders import read_abstract_ipc, read_fern, read_fern_categories
from meeg_loaders import read_dirani_n400
from behav_loaders import read_italian_behav, read_german_behav
from tms_loaders import read_it_distr_learn_tms, read_de_sound_act_tms, read_de_sem_phon_tms
from simrel_norms_loaders import read_men, read_simlex, read_ws353
from psycholing_norms_loaders import load_lancaster_en_de_it

from count_utils import build_ppmi_vecs, read_mitchell_25dims, load_count_coocs
from test_utils import args, load_dataset, load_static_model, test_model, test_count_model

args = args()
lancaster_ratings, trans_from_en = load_lancaster_en_de_it()

rows, datasets = load_dataset(args)
model, vocab = load_model(args)

### for static models, we only test once
static_models = [
                 'fasttext',
                 'fasttext_aligned',
                 'conceptnet',
                 ]
if args.model in static_models:
    model, vocab = load_static_model(args)
    test_model(
               args.lang, 
               args.model, 
               model, 
               vocab, 
               datasets, 
               trans_from_en,
               )
### for count models, we test with a lot of different possibilities
else:
    vocab, coocs, freqs = load_count_coocs(args)
    ### keeping row words that are actually available
    row_words = [w for w in rows[args.lang] if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0]
    #
    ### mitchell hand-picked dimensions
    #
    for row_mode in [
                     '', 
                     #'_rowincol',
                     ]:
        key = 'ppmi_{}_mitchell{}_words'.format(args.model, row_mode)
        if row_mode == 'rowincol':
            ctx_words = set([w for ws in read_mitchell_25dims(args.lang) for w in ws] + row_words)
        else:
            ctx_words = [w for ws in read_mitchell(args.lang) for w in ws]
        test_count_model(args, key, datasets, trans_from_en, coocs, vocab, row_words, ctx_words)
    #
    ### lancaster
    #
    filt_ratings = {w : freqs[w] for w in lancaster_ratings[lang].keys() if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0}
    sorted_ratings = [w[0] for w in sorted(filt_ratings.items(), key=lambda item: item[1], reverse=True)]
    filt_perc = {w : v['minkowski3'] for w, v in lancaster_ratings[lang].items() if w in vocab.keys() and w in freqs.keys() and vocab[w]!=0}
    sorted_perc = [w[0] for w in sorted(filt_perc.items(), key=lambda item: item[1], reverse=True)]
    for freq in tqdm([
                      100, 
                      200, 
                      500, 
                      750,
                      1000, 
                      2500, 
                      5000, 
                      7500,
                      10000, 
                      12500, 
                      15000, 
                      17500,
                      20000, 
                      25000,
                      ]):
        for row_mode in [
                         '', 
                         #'_rowincol',
                         ]:
            for selection_mode in [
                                   'top', 
                                   #'random', 
                                   'hi-perceptual', 
                                   'lo-perceptual',
                                   ]: 
                key = 'ppmi_{}_lancaster_freq_{}{}_{}_words'.format(args.model, selection_mode, row_mode, freq)
                if selection_mode == 'top':
                    if row_mode == 'rowincol':
                        ctx_words = set([w for w in sorted_ratings[:freq]]+row_words)
                    else:
                        ctx_words = [w for w in sorted_ratings[:freq]]
                elif selection_mode == 'hi-perceptual':
                    if row_mode == 'rowincol':
                        ctx_words = set([w for w in sorted_perc[:freq]]+row_words)
                    else:
                        ctx_words = [w for w in sorted_perc[:freq]]
                elif selection_mode == 'lo-perceptual':
                    if row_mode == 'rowincol':
                        ctx_words = set([w for w in sorted_perc[-freq:]]+row_words)
                    else:
                        ctx_words = [w for w in sorted_perc[-freq:]]
                else:
                    random.seed(12)
                    idxs = random.sample(range(len(sorted_ratings)), k=min(freq, len(sorted_ratings)))
                    if row_mode == 'rowincol':
                        ctx_words = set([sorted_ratings[i] for i in idxs]+row_words)
                    else:
                        ctx_words = [sorted_ratings[i] for i in idxs]
                test_count_model(args, key, datasets, trans_from_en, coocs, vocab, row_words, ctx_words)
                #ctx_idxs = [vocab[w] for w in ctx_words]
                #trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, row_words, ctx_words, smoothing=False)
                #model = {k : v for k, v in trans_pmi_vecs.items()}
                #curr_vocab = [w for w in trans_pmi_vecs.keys()]
                #test_model(lang, key, model, curr_vocab, datasets, trans_from_en)
    #
    ### top-n frequencies
    #
    filt_freqs = {w : f for w, f in freqs.items() if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0}
    sorted_freqs = [w[0] for w in sorted(filt_freqs.items(), key=lambda item: item[1], reverse=True)]
    for freq in tqdm([
                      100, 
                      200, 
                      500, 
                      750,
                      1000, 
                      2500, 
                      5000, 
                      7500,
                      10000, 
                      12500, 
                      15000, 
                      17500,
                      20000, 
                      25000,
                      ]):
        for row_mode in [
                         '_', 
                         #'_rowincol',
                         ]:
            for selection_mode in [
                                   'top', 
                                   #'random',
                                   ]: 
                key = 'ppmi_{}_abs_freq_{}{}_{}_words'.format(args.model, selection_mode, row_mode, freq)
                if selection_mode == 'top':
                    if row_mode == 'rowincol':
                        ctx_words = set([w for w in sorted_freqs[:freq]]+row_words)
                    else:
                        ctx_words = [w for w in sorted_freqs[:freq]]
                else:
                    random.seed(12)
                    idxs = random.sample(range(len(sorted_freqs)), k=min(freq, len(sorted_freqs)))
                    if row_mode == 'rowincol':
                        ctx_words = set([sorted_freqs[i] for i in idxs]+row_words)
                    else:
                        ctx_words = [sorted_freqs[i] for i in idxs]
                ### using the basic required vocab for all tests as a basis set of words
                test_count_model(args, key, datasets, trans_from_en, coocs, vocab, row_words, ctx_words)
                #ctx_idxs = [vocab[w] for w in ctx_words]
                #trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, row_words, ctx_words, smoothing=False)
                #model = {k : v for k, v in trans_pmi_vecs.items()}
                #curr_vocab = [w for w in trans_pmi_vecs.keys()]
                #test_model(lang, key, model, curr_vocab, datasets, trans_from_en)
