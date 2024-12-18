import fasttext
import numpy
import os
import pickle
import scipy
import sys

from tqdm import tqdm

sys.path.append('..')

from utf_utils import transform_german_word
from utils import build_ppmi_vecs, read_lancaster_ratings

lancaster_ratings = {
                     'en' : read_lancaster_ratings(),
                     }
### german translation
for lang in ['de', 'it']:
    lancaster_ratings[lang] = dict()
    with tqdm() as counter:
        with open(os.path.join('..', 'translations', 'lanc_fern_{}_to_en.tsv'.format(lang))) as i:
            for l in i:
                line = l.strip().split('\t')
                if lang == 'de':
                    for w in transform_german_word(line[0].lower()):
                        try:
                            lancaster_ratings['de'][w] = lancaster_ratings['en'][line[1]]
                            counter.update(1)
                        except KeyError:
                            continue
                elif lang == 'it':
                    for w in [line[0].lower(), line[0].capitalize()]:
                        try:
                            lancaster_ratings['it'][w] = lancaster_ratings['en'][line[1]]
                            counter.update(1)
                        except KeyError:
                            continue

lang = 'it'
###italian cerebellum tms
model = dict()
ft = fasttext.load_model('../../../dataset/word_vectors/{}/cc.{}.300.bin'.format(lang, lang))
model['ft'] = ft
other_model = 'ppmi'
lines = list()
row_words = list()
with open(os.path.join('..', 'tms', 'data', 'italian_tms_cereb.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = [w for w in line]
            continue
        ### adding words
        row_words.append(line[header.index('noun')].lower())
        row_words.append(line[header.index('noun')].capitalize())
        row_words.append(line[header.index('adj')].lower())
        row_words.append(line[header.index('adj')].capitalize())
        lines.append(line)
row_words = set(row_words)
corpus = 'cc100'
min_count = 100
with open(os.path.join(
                       '..', '..', 'psychorpus',
                       'pickles', lang, corpus, 
                       '{}_{}_uncased_vocab_min_{}.pkl'.format(lang, corpus, min_count),
                       ), 'rb') as i:
    vocab = pickle.load(i)
with open(os.path.join(
                       '..', '..', 'psychorpus',
                       'pickles', lang, corpus, 
                       '{}_{}_uncased_word_freqs.pkl'.format(lang, corpus),
                       ), 'rb') as i:
    freqs = pickle.load(i)
print('total size of the corpus: {:,} tokens'.format(sum(freqs.values())))
with open(os.path.join(
                       '..', '..', 'psychorpus',
                       'pickles', lang, corpus, 
                       '{}_{}_coocs_uncased_min_{}_win_4.pkl'.format(lang, corpus, min_count),
                       ), 'rb') as i:
    coocs = pickle.load(i)
row_words = [w for w in row_words if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0]
### lancaster
filt_ratings = {w : freqs[w] for w in lancaster_ratings[lang].keys() if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0}
sorted_ratings = [w[0] for w in sorted(filt_ratings.items(), key=lambda item: item[1], reverse=True)]
filt_perc = {w : v['minkowski3'] for w, v in lancaster_ratings[lang].items() if w in vocab.keys() and w in freqs.keys() and vocab[w]!=0}
sorted_perc = [w[0] for w in sorted(filt_perc.items(), key=lambda item: item[1], reverse=True)]
for freq in tqdm([
                  5000
                  ]):
    for row_mode in [
                     '', 
                     ]:
        for selection_mode in [
                               'top', 
                               ]: 
            key = 'ppmi_{}_lancaster_freq_{}_{}_{}_words'.format(corpus, selection_mode, row_mode, freq)
            if selection_mode == 'top':
                ctx_words = [w for w in sorted_ratings[:freq]]
            elif selection_mode == 'hi-perceptual':
                ctx_words = [w for w in sorted_perc[:freq]]
            elif selection_mode == 'lo-perceptual':
                ctx_words = [w for w in sorted_perc[-freq:]]
            ctx_idxs = [vocab[w] for w in ctx_words]
            trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, row_words, ctx_words, smoothing=False)
            model[other_model] = {k : trans_pmi_vecs[k] for k in row_words} 

out = 'out'
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, 'it_tms_cerebellum_r.tsv'), 'w') as o:
    o.write(
            'condition\t'\
            'dummy_condition\t'\
            'contrast_condition\t'\
            'timing\t'\
            'dummy_timing\t'\
            'contrast_timing\t'\
            'subject\t'\
            'noun\t'\
            'adj\t'\
            'item\t'\
            'rt\t'\
            'log_rt\t'\
            'accuracy\t'\
            'original_similarity\t'\
            'fasttext_cos_similarity\t'\
            '{}_cos_similarity\n'.format(other_model)\
            )
    for l in lines:
        cond = l[header.index('condition')]
        dummy_cond = 0. if cond == 'cz' else 1.
        contrast_cond = -.5 if cond == 'cz' else .5
        rt = float(l[header.index('RTs')].replace(',', '.'))
        original_sim = 1 - float(l[header.index('DistSim')].replace(',', '.'))
        log_rt = numpy.log10(rt)
        sub = l[header.index('Subject')]
        noun = l[header.index('noun')]
        adj = l[header.index('adj')]
        item = '{}_{}'.format(noun, adj)
        #order = l[header.index('order')]
        accuracy = l[header.index('accuracy')]
        timing = l[header.index('TMS_Timing')]
        dummy_timing = 0. if timing == 'noun' else 1.
        contrast_timing = -.5 if timing == 'noun' else .5
        o.write(
            '{}\t'.format(cond)+''\
            '{}\t'.format(dummy_cond)+''\
            '{}\t'.format(contrast_cond)+''\
            '{}\t'.format(timing)+''\
            '{}\t'.format(dummy_timing)+''\
            '{}\t'.format(contrast_timing)+''\
            '{}\t'.format(sub)+''\
            '{}\t'.format(noun)+''\
            '{}\t'.format(item)+''\
            '{}\t'.format(adj)+''\
            '{}\t'.format(rt)+''\
            '{}\t'.format(log_rt)+''\
            '{}\t'.format(accuracy)+''\
            '{}\t'.format(original_sim)+''\
            )
        for model_name in ['ft', other_model]:
            sims = list()
            for w_one in [noun.lower(), noun.capitalize()]:
                for w_two in [adj.lower(), adj.capitalize()]:
                    try:
                        sim = 1 - scipy.spatial.distance.cosine(model[model_name][w_one], model[model_name][w_two])
                        sims.append(sim)
                    except KeyError:
                        continue
            if len(sims) > 0:
                final_sim = numpy.average(sims)
            else:
                final_sim = 'NA'
            if model_name == 'ft':
                o.write('{}\t'.format(final_sim))
            else:
                o.write('{}\n'.format(final_sim))
lang = 'de'
###german ifg tms
de_data = dict()
row_words = list()
lines = list()
with open(os.path.join('..', 'tms', 'data', 'german_ifg_r.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = [w for w in line]
            lines.append(header)
        else:
            ### adding words
            c = line[header.index('category')].lower()
            for w in transform_german_word(c):
                row_words.append(w)
            w = line[header.index('word')].lower()
            for t_w in transform_german_word(w):
                row_words.append(t_w)
            lines.append(line)
    de_data['ifg'] = [l for l in lines]
###german pIPL tms
lines = list()
with open(os.path.join('..', 'tms', 'data', 'de_tms_pipl.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = [w for w in line]
            lines.append(header)
        else:
            ### adding words
            if line[header.index('task')] == 'lexical_decision':
                continue
            t = line[header.index('task')].lower()
            for w in transform_german_word(t):
                row_words.append(w)
            s = line[header.index('stimulus')].lower()
            for w in transform_german_word(s):
                row_words.append(w)
            lines.append(line)
    de_data['pipl'] = [l for l in lines]
model = dict()
ft = fasttext.load_model('../../../dataset/word_vectors/{}/cc.{}.300.bin'.format(lang, lang))
model['ft'] = ft
other_model = 'ppmi'
row_words = set(row_words)
corpus = 'cc100'
min_count = 100
with open(os.path.join(
                       '..', '..', 'psychorpus',
                       'pickles', lang, corpus, 
                       '{}_{}_uncased_vocab_min_{}.pkl'.format(lang, corpus, min_count),
                       ), 'rb') as i:
    vocab = pickle.load(i)
with open(os.path.join(
                       '..', '..', 'psychorpus',
                       'pickles', lang, corpus, 
                       '{}_{}_uncased_word_freqs.pkl'.format(lang, corpus),
                       ), 'rb') as i:
    freqs = pickle.load(i)
print('total size of the corpus: {:,} tokens'.format(sum(freqs.values())))
with open(os.path.join(
                       '..', '..', 'psychorpus',
                       'pickles', lang, corpus, 
                       '{}_{}_coocs_uncased_min_{}_win_4.pkl'.format(lang, corpus, min_count),
                       ), 'rb') as i:
    coocs = pickle.load(i)
row_words = [w for w in row_words if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0]
### lancaster
filt_ratings = {w : freqs[w] for w in lancaster_ratings[lang].keys() if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0}
sorted_ratings = [w[0] for w in sorted(filt_ratings.items(), key=lambda item: item[1], reverse=True)]
filt_perc = {w : v['minkowski3'] for w, v in lancaster_ratings[lang].items() if w in vocab.keys() and w in freqs.keys() and vocab[w]!=0}
sorted_perc = [w[0] for w in sorted(filt_perc.items(), key=lambda item: item[1], reverse=True)]
for freq in tqdm([
                  5000
                  ]):
    for row_mode in [
                     '', 
                     ]:
        for selection_mode in [
                               'top', 
                               ]: 
            key = 'ppmi_{}_lancaster_freq_{}_{}_{}_words'.format(corpus, selection_mode, row_mode, freq)
            if selection_mode == 'top':
                ctx_words = [w for w in sorted_ratings[:freq]]
            elif selection_mode == 'hi-perceptual':
                ctx_words = [w for w in sorted_perc[:freq]]
            elif selection_mode == 'lo-perceptual':
                ctx_words = [w for w in sorted_perc[-freq:]]
            ctx_idxs = [vocab[w] for w in ctx_words]
            trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, row_words, ctx_words, smoothing=False)
            model[other_model] = {k : trans_pmi_vecs[k] for k in row_words} 

out = 'out'
os.makedirs(out, exist_ok=True)
for area, data in de_data.items():
    with open(os.path.join(out, 'de_tms_{}.tsv'.format(area)), 'w') as o:
        if area == 'ifg':
            o.write(
                    'condition\t'\
                    'dummy_condition\t'\
                    'subject\t'\
                    'session\t'\
                    'category\t'\
                    'word\t'\
                    'rt\t'\
                    'log_rt\t'\
                    'fasttext_cos_similarity\t'\
                    '{}_cos_similarity\n'.format(other_model)\
                    )
            for l_i, l in enumerate(data):
                if l_i == 0:
                    header = [w for w in l]
                    print(header)
                    continue
                contrast_cond = l[header.index('condition')]
                if float(contrast_cond) == 1.:
                    cond = 'sham'
                if float(contrast_cond) == .5:
                    cond = 'aIFG'
                if float(contrast_cond) == 1.5:
                    cond = 'pIFG'
                rt = float(l[header.index('rt')])
                log_rt = float(l[header.index('log_rt')])
                sub = l[header.index('sub')]
                ses = l[header.index('session')]
                cat = l[header.index('category')]
                w = l[header.index('word')]
                o.write(
                    '{}\t'.format(cond)+''\
                    '{}\t'.format(contrast_cond)+''\
                    '{}\t'.format(sub)+''\
                    '{}\t'.format(ses)+''\
                    '{}\t'.format(cat)+''\
                    '{}\t'.format(w)+''\
                    '{}\t'.format(rt)+''\
                    '{}\t'.format(log_rt)+''\
                    )
                for model_name in ['ft', other_model]:
                    sims = list()
                    for w_one in transform_german_word(cat):
                        for w_two in transform_german_word(w):
                            try:
                                sim = 1 - scipy.spatial.distance.cosine(model[model_name][w_one], model[model_name][w_two])
                                sims.append(sim)
                            except KeyError:
                                continue
                    if len(sims) > 0:
                        final_sim = numpy.average(sims)
                    else:
                        final_sim = 'NA'
                    if model_name == 'ft':
                        o.write('{}\t'.format(final_sim))
                    else:
                        o.write('{}\n'.format(final_sim))
        if area == 'pipl':
            o.write(
                    'condition\t'\
                    'dummy_condition\t'\
                    'contrast_condition\t'\
                    'subject\t'\
                    'action_word\t'\
                    'sound_word\t'\
                    'task\t'\
                    'word\t'\
                    'accuracy\t'\
                    'rt\t'\
                    'log_rt\t'\
                    'fasttext_cos_similarity\t'\
                    '{}_cos_similarity\n'.format(other_model)\
                    )
            for l_i, l in enumerate(data):
                if l_i == 0:
                    header = [w for w in l]
                    print(header)
                    continue
                cond = l[header.index('condition')]
                dummy_cond = 0. if cond=='sham' else 1.
                contrast_cond = -.5 if cond=='sham' else .5
                exp = l[header.index('expected_response')]
                resp = l[header.index('response')]
                a_w = l[header.index('action_word')]
                s_w = l[header.index('sound_word')]
                acc = 1. if exp==resp else 0.
                rt = float(l[header.index('rt')])
                log_rt = float(l[header.index('log_rt')])
                sub = l[header.index('subject')]
                task = l[header.index('task')]
                w = l[header.index('stimulus')]
                o.write(
                    '{}\t'.format(cond)+''\
                    '{}\t'.format(dummy_cond)+''\
                    '{}\t'.format(contrast_cond)+''\
                    '{}\t'.format(sub)+''\
                    '{}\t'.format(a_w)+''\
                    '{}\t'.format(s_w)+''\
                    '{}\t'.format(task)+''\
                    '{}\t'.format(w)+''\
                    '{}\t'.format(acc)+''\
                    '{}\t'.format(rt)+''\
                    '{}\t'.format(log_rt)+''\
                    )
                for model_name in ['ft', other_model]:
                    sims = list()
                    for w_one in transform_german_word(cat):
                        for w_two in transform_german_word(w):
                            try:
                                sim = 1 - scipy.spatial.distance.cosine(model[model_name][w_one], model[model_name][w_two])
                                sims.append(sim)
                            except KeyError:
                                continue
                    if len(sims) > 0:
                        final_sim = numpy.average(sims)
                    else:
                        final_sim = 'NA'
                    if model_name == 'ft':
                        o.write('{}\t'.format(final_sim))
                    else:
                        o.write('{}\n'.format(final_sim))
