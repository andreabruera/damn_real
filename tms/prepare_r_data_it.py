import fasttext
import numpy
import os
import pickle
import scipy

from scipy import spatial
from tqdm import tqdm

from utils import build_ppmi_vecs, read_ratings
from utf_utils import transform_german_word

### loading german
ft = fasttext.load_model('../../dataset/word_vectors/it/cc.it.300.bin')
### loading frequencies
with open(os.path.join(
                       'pickles', 'it', 'joint', 
                       'it_joint_uncased_word_freqs.pkl',
                       ), 'rb') as i:
    freqs = pickle.load(i)
### loading pos
with open(os.path.join(
                       'pickles', 'it', 'joint', 
                       'it_joint_uncased_word_pos.pkl',
                       ), 'rb') as i:
    pos = pickle.load(i)
### vocab
with open(os.path.join(
                       'pickles', 'it', 'joint', 
                       'it_joint_uncased_vocab_min_10.pkl',
                       ), 'rb') as i:
    vocab = pickle.load(i)
### coocs
with open(os.path.join(
                       'pickles', 'it', 'joint', 
                       'it_joint_coocs_uncased_min_10_win_4.pkl',
                       ), 'rb') as i:
    coocs = pickle.load(i)

uttered_words = list()

with open(os.path.join(
                       'data', 
                       'italian_tms_cereb.tsv'
                       )) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            continue
        word = line[header.index('noun')].strip()
        if word == 'NA':
            continue
        w_ones = transform_german_word(word)
        w_ones = [w for w in w_ones if w in vocab.keys() and w in freqs.keys() and vocab[w]!=0]
        if len(w_ones) == 0:
            continue
        uttered_words.extend(w_ones)
        category = line[header.index('adj')].split('.')[0]
        w_twos = transform_german_word(category)
        w_twos = [w for w in w_twos if w in vocab.keys() and w in freqs.keys() and vocab[w]!=0]
        assert len(w_twos) > 0
        uttered_words.extend(w_twos)

ratings = read_ratings(hand=True)
test_words = [w for w in ratings.keys()]
trans = dict()
### reading translations
inv_trans = dict()
to_write = list()
fs = [
        #'en_to_de.tsv'
        'lancaster_english_to_german.tsv',
        'fernandino_english_to_german.tsv',
        ]
for f in fs:
    with open(os.path.join('data', f)) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.lower().strip().split('\t')
            trans[line[0].strip()] = line[1].strip().replace('-', '_').lower()
            inv_trans[line[1].strip().replace('-', '_').lower()] = line[0].strip()
            to_write.append((line[0].strip(), line[1].strip().replace('-', '_').lower()))
with open(os.path.join('data', 'en_to_de.tsv'), 'w') as o:
    o.write('english\tgerman\n')
    for k, v in to_write:
        o.write('{}\t{}\n'.format(k, v))
missing = list()
present = dict()
for w in test_words:
    try:
        w = trans[w].lower()
        #if inv_trans[trans[w].lower()] != w:
        #    missing.append(w)
        #    continue
    except KeyError:
        missing.append(w)
        continue
    if w not in freqs.keys():
        missing.append(w)
        print('{} - not appearing at all'.format(w))
        continue
    if w not in vocab.keys() or vocab[w]==0:
        missing.append(w)
        continue
    if vocab[w] not in coocs.keys():
        missing.append(w)
        print([w, freqs[w]])
        continue
    else:
        present[w] = freqs[w]
print('\n')
print('number of words missing: {}'.format(len(missing)))
print('missing words:')
#print(missing)
min_n = min(list(present.values()))
max_n = max(list(present.values()))
lower_ten_ws = [w for w, val in present.items() if val < 10]
lower_fifty_ws = [w for w, val in present.items() if val < 50]
lower_hundred_ws = [w for w, val in present.items() if val < 100]
ten_n = sum([1 for val in present.values() if val >= 10])
fifty_n = sum([1 for val in present.values() if val >= 50])
hundred_n = sum([1 for val in present.values() if val >= 100])
avg_n = numpy.average(list(present.values()))
med_n = numpy.median(list(present.values()))
print('minimum number of mentions: {}'.format(min_n))
print('maximum number of mentions: {}'.format(max_n))
print('number of items above 10 mentions: {}'.format(ten_n))
print('items with less than 10 mentions:')
#print(lower_ten_ws)
print('number of items above 50 mentions: {}'.format(fifty_n))
print('items with less than 50 mentions:')
#print(lower_fifty_ws)
print('number of items above 100 mentions: {}'.format(hundred_n))
print('items with less than 100 mentions:')
#print(lower_hundred_ws)
print('average number of mentions: {}'.format(avg_n))
print('median number of mentions: {}'.format(med_n))

#pruned_test_words = [w for w in uttered_words if w not in missing and vocab[w]!=0]
pruned_test_words = [w for w in present.keys()]
pruned_ratings = {trans[w] : dct for w, dct in ratings.items() if trans[w] in present.keys()}
        #trans.keys() and trans[w] in freqs.keys() and vocab[trans[w]]!=0 and vocab[trans[w]] in coocs.keys()}
ctx_words = set(pruned_test_words)
mtrx_ctx_words = sorted(ctx_words.union(set(uttered_words)))
mtrx_ctx_idxs = [vocab[w] for w in ctx_words]
sem_dims = set([var for k,v in pruned_ratings.items() for var in v.keys()])

### damaging
damaged_ppmi = dict()
for orig_percent in tqdm([
                     0, 
                     10, 20, 
                     30, 
                     40, 50, 60,
                     ]):
    ### 0.002 because there are 5 senses (0.01/5=0.002)
    percent_damage = orig_percent*0.002
    percent = int(len(pruned_ratings.items())*percent_damage)
    if orig_percent == 0:
        #vecs = {w : numpy.array([coocs[vocab[w]][idx] if idx in coocs[vocab[w]].keys() else 0 for idx in mtrx_ctx_idxs]) for w in pruned_test_words}
        undam_trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, mtrx_ctx_words, mtrx_ctx_words, smoothing=False)
        undam_ws = sorted(undam_trans_pmi_vecs.keys())
        undam_mtrx = numpy.array([undam_trans_pmi_vecs[k] for k in undam_ws])
        damaged_ppmi[orig_percent] = {k : undam_trans_pmi_vecs[k] for k in uttered_words}
        uttered_idxs = [mtrx_ctx_words.index(w) for w in uttered_words]
    else:
        dam_ctx_words = list()
        for dim in sem_dims:
            if dim in ['concreteness', 'hand_arm']:
                continue
            sorted_ws = sorted([(w, v[dim]) for w, v in pruned_ratings.items()], key=lambda item: item[1])
            dam_ctx_words.extend([w for w, val in sorted_ws[-percent:]])
            ### also adding super abstract words
            #    ctx_words = ctx_words.union(set([w[0] for w in sorted_ws[:one_percent]]))
            #    ctx_words = ctx_words.union(set([w for w, val in sorted_ws[-one_percent:]]))
        dam_ctx_words = sorted(set(dam_ctx_words))
        dam_idxs = [mtrx_ctx_words.index(w) for w in dam_ctx_words]
        dam_mtrx = numpy.copy(undam_mtrx)
        for idx in dam_idxs:
            dam_mtrx[:, idx][dam_idxs] = 0.
            ### columns
            dam_mtrx[idx, :][dam_idxs] = 0.
            ### our words
            #dam_mtrx[idx, :][uttered_idxs] = 0.
        #for idx in uttered_idxs:
        #    dam_mtrx[:, idx][dam_idxs] = 0.
        ### pmi
        ### building the PPMI matrix
        damaged_ppmi[orig_percent] = {k : dam_mtrx[mtrx_ctx_words.index(k)] for k in uttered_words}

### german IFG
#uttered_words = set(uttered_words)
with open('german_ifg_r.tsv', 'w') as o:
    o.write('sub\t')
    o.write('session\t')
    o.write('condition\t')
    o.write('category\t')
    o.write('word\t')
    o.write('rt\t')
    o.write('log_rt\t')
    o.write('ft_distance\t')
    for k in sorted(damaged_ppmi.keys()):
        o.write('ppmi_distance_damage_{}\t'.format(k))
    o.write('w_one_frequency\t')
    o.write('w_one_log10_frequency\t')
    o.write('w_one_length\t')
    o.write('w_two_frequency\t')
    o.write('w_two_log10_frequency\t')
    o.write('w_two_length\t')
    o.write('\n')
    with open(os.path.join(
                           'data', 
                           'german_ifg.tsv'
                           )) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = line.copy()
                continue
            word = line[header.index('noun')].strip()
            if word == 'NA':
                continue
            w_ones = transform_german_word(word)
            #print(w_ones)
            w_ones = [w for w in w_ones if w in freqs.keys() and vocab[w]!=0]
            if len(w_ones) == 0:
                continue
            freq_w_one = sum([freqs[w] for w in w_ones])
            len_w_one = len(word)
            #if word.lower() not in vocab.keys() or vocab[word]==0:
            #    print(word)
            sub = line[header.index('subj')]
            category = line[header.index('adj')].split('.')[0]
            w_twos = transform_german_word(category)
            w_twos = [w for w in w_twos if w in freqs.keys() and vocab[w]!=0]
            assert len(w_twos) > 0
            freq_w_two = sum([freqs[w] for w in w_twos])
            len_w_two = len(category)
            session = line[header.index('session')]
            condition = line[header.index('stim')]
            ### dummy coding, reference at 1.
            if condition == 'vertex':
                condition = 1.
            elif condition == 'aIFG':
                condition = 1.5
            elif condition == 'pIFG':
                condition = 0.5
            else:
                raise RuntimeError()
            rt = float(line[header.index('RT')].replace(',', '.'))
            ### rt
            log_rt = numpy.log(1+rt)
            ### ft
            combs = list()
            for w_one in w_ones:
                for w_two in w_twos:
                    combs.append((w_one, w_two))
            ft_distance = list()
            ppmi_distance = dict()
            for w_one, w_two in combs:
                ft_distance.append(scipy.spatial.distance.cosine(ft[w_one], ft[w_two]))
                for k, model in damaged_ppmi.items():
                    if k not in ppmi_distance.keys():
                        ppmi_distance[k] = list()
                    ppmi_distance[k].append(scipy.spatial.distance.cosine(model[w_one], model[w_two]))
            ft_distance = numpy.average(ft_distance)
            ppmi_distance = {k : numpy.average(v) for k, v in ppmi_distance.items()}
            o.write('{}\t'.format(sub))
            o.write('{}\t'.format(session))
            o.write('{}\t'.format(condition))
            o.write('{}\t'.format(category))
            o.write('{}\t'.format(word))
            o.write('{}\t'.format(rt))
            o.write('{}\t'.format(log_rt))
            o.write('{}\t'.format(ft_distance))
            for k in sorted(damaged_ppmi.keys()):
                o.write('{}\t'.format(ppmi_distance[k]))
            o.write('{}\t'.format(freq_w_one))
            o.write('{}\t'.format(numpy.log10(freq_w_one)))
            o.write('{}\t'.format(len(word)))
            o.write('{}\t'.format(freq_w_two))
            o.write('{}\t'.format(numpy.log10(freq_w_two)))
            o.write('{}\t'.format(len(category)))
            o.write('\n')
            #print(word)
