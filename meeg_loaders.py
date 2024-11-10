import numpy
import os

from utf_utils import transform_german_word, transform_basic_word

def read_dirani_n400(args):
    dis_sims = {'{}_dirani-n400_{}#words'.format(args.lang, args.stat_approach) : dict(), '{}_dirani-n400_{}#pictures'.format(args.lang, args.stat_approach) : dict()}
    if args.lang != 'en':
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
                if args.lang != 'en':
                    w_one = trans[w_one]
                    w_two = trans[w_two]
                ### german will be transformed
                if args.lang != 'de':
                    test_vocab = test_vocab.union(
                                                  set(
                              transform_basic_word(w_one)+transform_basic_word(w_two)
                                                      )
                                                  )
                else:
                    test_vocab = test_vocab.union(transform_german_word(w_one))
                    test_vocab = test_vocab.union(transform_german_word(w_two))
                #sim = numpy.average(numpy.array(line[2:], dtype=numpy.float32))
                ### we use dissimilarity!
                for sub, sim in enumerate(line[2:]):
                    if sub not in dis_sims[dataset].keys():
                        dis_sims[dataset][sub] = list()
                    dis_sims[dataset][sub].append((
                                                   (w_one, w_two), 
                                                   1 - float(sim)
                                                   ))

    return dis_sims, test_vocab

def read_kaneshiro_n400(args):
    dis_sims = {'{}_kaneshiro-n400_{}#pictures'.format(args.lang, args.stat_approach) : dict()}
    if args.lang != 'en':
        trans_path = os.path.join(
                                 'data', 
                                 'meeg',
                                 'kaneshiro2015-eeg_pairwise',
                                 'kaneshiro2015_en_it_de.tsv'
                                 )
        trans = dict()
        with open(trans_path) as i:
            for l_i, l in enumerate(i):
                line = [w.strip().lower() for w in l.split('\t')]
                if l_i == 0:
                    rel_trans = line.index(args.lang)
                    continue
                trans[line[0].strip()] = line[rel_trans].strip()
        print(trans)

    test_vocab = set()
    for dataset in dis_sims.keys():
        file_path = os.path.join(
                                 'data', 
                                 'meeg',
                                 'kaneshiro2015-eeg_pairwise',
                                 'tsv',
                                 '300-400ms_similarities.tsv')
        with open(file_path) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = [w.strip().lower() for w in l.strip().split('\t')]
                w_one = line[0].split('_')[1]
                w_two = line[1].split('_')[1]
                ### translating if needed
                if args.lang != 'en':
                    w_one = trans[w_one]
                    trans_w_one = '{}_{}'.format(line[0].split('_')[0], w_one)
                    w_two = trans[w_two]
                    trans_w_two = '{}_{}'.format(line[1].split('_')[0], w_two)
                ### german will be transformed
                if args.lang != 'de':
                    test_vocab = test_vocab.union(
                                                  set(
                              transform_basic_word(w_one)+transform_basic_word(w_two)
                                                      )
                                                  )
                else:
                    test_vocab = test_vocab.union(transform_german_word(w_one))
                    test_vocab = test_vocab.union(transform_german_word(w_two))
                #sim = numpy.average(numpy.array(line[2:], dtype=numpy.float32))
                ### we use dissimilarity!
                for sub, sim in enumerate(line[2:]):
                    if sub not in dis_sims[dataset].keys():
                        dis_sims[dataset][sub] = list()
                    if args.lang != 'en':
                        dis_sims[dataset][sub].append((
                                                      (
                                                       trans_w_one.split('_')[-1], 
                                                       trans_w_two.split('_')[-1],
                                                       ),  
                                                      1 - float(sim)
                                                      ))
                    else:
                        dis_sims[dataset][sub].append((
                                                      (
                                                       line[0].split('_')[-1], 
                                                       line[1].split('_')[-1],
                                                       ), 
                                                      1 - float(sim)
                                                      ))

    return dis_sims, test_vocab
