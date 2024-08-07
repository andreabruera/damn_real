import numpy
import os
import pickle
import random

def read_mitchell_25dims(lang):
    assert lang == 'en'
    dimensions = list()
    with open(os.path.join('data', 'fmri', 'mitchell', 'mitchell_dimensions.tsv')) as i:
        for l in i:
            line = l.strip().split()
            assert len(line) >= 2
            dimensions.append(line)
    assert len(dimensions) == 25

    return dimensions

def build_ppmi_vecs(coocs, vocab, row_words, col_words, smoothing=False, power=1.):
    pmi_mtrx = numpy.array(
                             [
                              [coocs[vocab[w]][vocab[w_two]] if vocab[w_two] in coocs[vocab[w]].keys() else 0 for w_two in col_words]
                              for w in row_words])
    assert pmi_mtrx.shape[0] == len(row_words)
    assert pmi_mtrx.shape[1] == len(col_words)
    if power != 1.:
        pmi_mtrx = numpy.power(pmi_mtrx, power)
    #matrix_[matrix_ != 0] = np.array(1.0/matrix_[matrix_ != 0])
    axis_one_sum = pmi_mtrx.sum(axis=1)
    #axis_one_mtrx = numpy.divide(1, axis_one_sum, where=axis_one_sum!=0).reshape(-1, 1)
    axis_one_mtrx = numpy.array([1/val if val!=0 else val for val in axis_one_sum]).reshape(-1, 1)
    assert True not in numpy.isnan(axis_one_mtrx)
    axis_zero_sum = pmi_mtrx.sum(axis=0)
    #axis_zero_mtrx = numpy.divide(1, axis_zero_sum, where=axis_zero_sum!=0).reshape(1, -1)
    axis_zero_mtrx = numpy.array([1/val if val!=0 else val for val in axis_zero_sum]).reshape(1, -1)
    assert True not in numpy.isnan(axis_one_mtrx)
    ### raising to 0.75 as suggested in Levy & Goldberg 2015
    if smoothing:
        total_sum = numpy.power(pmi_mtrx, 0.75).sum()
    else:
        total_sum = pmi_mtrx.sum()
    #trans_pmi_mtrx = numpy.multiply(numpy.multiply(numpy.multiply(pmi_mtrx,1/pmi_mtrx.sum(axis=1).reshape(-1, 1)), 1/pmi_mtrx.sum(axis=0).reshape(1, -1)), pmi_mtrx.sum())
    trans_pmi_mtrx = numpy.multiply(
                                    numpy.multiply(
                                                   numpy.multiply(
                                                                  pmi_mtrx,axis_one_mtrx), 
                                                   axis_zero_mtrx), 
                                    total_sum)
    trans_pmi_mtrx[trans_pmi_mtrx<1.] = 1
    assert True not in numpy.isnan(trans_pmi_mtrx.flatten())
    ### checking for nans
    trans_pmi_vecs = {w : numpy.log2(trans_pmi_mtrx[w_i]) for w_i, w in enumerate(row_words)}
    for v in trans_pmi_vecs.values():
        assert True not in numpy.isnan(v)

    return trans_pmi_vecs

def load_count_coocs(args):
    print(corpus)
    if args.lang == 'en':
        if args.model == 'bnc':
            min_count = 10
        elif model == 'cc100':
            min_count = 500
        else:
            min_count = 100
    else:
        if args.model == 'cc100':
            min_count = 100
        else:
            min_count = 10
    with open(os.path.join(
                            '/',
                            'data',
                            'tu_bruera',
                            'counts',
                           args.lang, 
                           args.model, 
                           '{}_{}_uncased_vocab_min_{}.pkl'.format(
                                                                   args.lang, 
                                                                   args.model, 
                                                                   min_count
                                                                   ),
                           ), 'rb') as i:
        vocab = pickle.load(i)
    with open(os.path.join(
                            '/',
                            'data',
                            #'u_bruera_software',
                            'tu_bruera',
                            'counts',
                           args.lang, 
                           args.model, 
                           '{}_{}_uncased_word_freqs.pkl'.format(
                                                                 args.lang, 
                                                                 args.model
                                                                 ),
                           ), 'rb') as i:
        freqs = pickle.load(i)
    print('total size of the corpus: {:,} tokens'.format(sum(freqs.values())))
    with open(os.path.join(
                            '/',
                            'data',
                            #'u_bruera_software',
                            'tu_bruera',
                            'counts',
                            args.lang, 
                            args.model, 
                           '{}_{}_coocs_uncased_min_{}_win_4.pkl'.format(
                                                                         args.lang,
                                                                         args.model, 
                                                                         min_count
                                                                         ),
                           ), 'rb') as i:
        coocs = pickle.load(i)
    return vocab, coocs, freqs
