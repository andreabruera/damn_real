import numpy
import os

def read_dirani_n400(args):
    dis_sims = {'{}_dirani-n400-words'.format(args.lang) : dict(), '{}_dirani-n400-pictures'.format(args.lang) : dict()}
    if lang != 'en':
        trans_path = os.path.join(
                                 'data', 
                                 'meeg',
                                 'dirani',
                                 'meg-dirani-n400_stimuli_translations_it_de.tsv'
                                 )
        trans = dict()
        with open(trans_path) as i:
            for l_i, l in enumerate(i):
                line = l.strip().split('\t')
                if l_i == 0:
                    rel_trans = line.index('word_{}'.format(args.lang))
                    continue
                trans[line[0].strip()] = line[rel_trans].strip()

    test_vocab = set()
    for dataset in dis_sims.keys():
        dis_sims[dataset] = dict()
        modality = 'words' if 'words' in dataset else 'pictures'
        file_path = os.path.join(
                                 'data', 
                                 'meeg',
                                 'dirani',
                                 'reorganized_dataset',
                                 modality,
                                 'tsv',
                                 '300-400ms_similarities.tsv')
        with open(file_path) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                w_one = line[0]
                w_two = line[1]
                ### translating if needed
                if lang != 'en':
                    w_one = trans[w_one]
                    w_two = trans[w_two]
                ### german will be transformed
                if lang != 'de':
                    test_vocab = test_vocab.union(set([w_one, w_two]))
                else:
                    test_vocab = test_vocab.union(transform_german_word(w_one))
                    test_vocab = test_vocab.union(transform_german_word(w_two))
                #sim = numpy.average(numpy.array(line[2:], dtype=numpy.float32))
                ### we use dissimilarity!
                for sub, sim in enumerate(line[2:]):
                    if sub not in dis_sims[dataset].keys():
                        dis_sims[dataset][sub] = dict()
                    dis_sims[dataset][sub][(w_one, w_two)] = 1 - sim

    return dis_sims, test_vocab
